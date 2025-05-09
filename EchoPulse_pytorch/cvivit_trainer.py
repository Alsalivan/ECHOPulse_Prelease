import pdb
from math import sqrt
from random import choice
from pathlib import Path
from shutil import rmtree
# import webdataset as wds
import torchvision.io
import io
import json
import os
import pickle
import re
import tempfile
from einops import rearrange
import time

from beartype import beartype

import torch
import glob
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

from einops import rearrange
from tqdm import tqdm

from EchoPulse_pytorch.optimizer import get_optimizer, LinearWarmup_CosineAnnealing

from ema_pytorch import EMA

from EchoPulse_pytorch.cvivit import CViViT
from EchoPulse_pytorch.data import VideoDatasetCMR, ImageDataset, VideoDataset, video_tensor_to_gif, video_to_tensor, video_tensor_to_pil_first_image

from accelerate import Accelerator

import wandb

# helpers


def exists(val):
    return val is not None


def cycle(dl):
    while True:
        for data in dl:
            yield data


def noop(*args, **kwargs):
    pass


def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)


def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# main trainer class


@beartype
class CViViTTrainer(nn.Module):
    def __init__(
        self,
        vae,
        *,
        accelerator,
        num_train_steps,
        batch_size,
        folder,
        num_frames=11,
        lr=3e-4,
        grad_accum_every=1,
        wd=0.,
        max_grad_norm=0.5,
        train_on_images=False,
        force_cpu=False,
        wandb_mode="disabled",
        discr_max_grad_norm=None,
        linear_warmup_start_factor=0.1,
        linear_warmup_total_iters=100,
        cosine_annealing_T_max=1000000,
        cosine_annealing_eta_min=1e-5,
        save_results_every=1000,
        save_model_every=1000,
        results_folder='./results',
        scheduler_optim_overhead=0,
        valid_frac=0.05,
        random_split_seed=42,
        use_ema=True,
        ema_beta=0.995,
        ema_update_after_step=0,
        ema_update_every=1,
        apply_grad_penalty_every=4,
        inference=False,
        log_every=10,
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        self.accelerator = accelerator
        
        # Initialize the attribute for custom objects
        self._custom_objects = []
        image_size = vae.image_size

        self.ds = VideoDatasetCMR(
            image_size=image_size,
            mode='train',
            folder=folder
        )
        self.valid_ds = VideoDatasetCMR(
            image_size=image_size,
            mode='val',
            folder=folder
        )
                
        # wandb config
        config = {}
        arguments = locals()
        for key in arguments.keys():
            if key not in ['self', 'config', '__class__', 'vae']:
                config[key] = arguments[key]
                
        config.update(vae.get_config())

        # 3. Log gradients and model parameters
        # if (wandb_mode != "disabled"):
        #    wandb.watch(vae, log='all', log_freq=3)

        if not inference:
            self.wandb_mode = wandb_mode
        else:
            self.wandb_mode = 'disabled'
            
        self.log_every = log_every

        self.accelerator.init_trackers(
            project_name="CViViT",
            config=config,
            init_kwargs={"wandb": {"mode": self.wandb_mode}}
        )

        if self.accelerator.is_main_process:
            print('config\n')
            print(config)
            
        self.vae = vae
        self.vae.wandb_mode = wandb_mode
        self.use_discr = vae.use_discr

        self.use_ema = use_ema
        if self.is_main and use_ema:
            self.ema_vae = EMA(
                vae, update_after_step=ema_update_after_step, update_every=ema_update_every)

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        all_parameters = list(vae.parameters())

        # Identify non-trainable (frozen or externally initialized) submodules
        non_vae_parameters = []
        if exists(vae.i3d):
            non_vae_parameters += list(vae.i3d.parameters())
        if exists(vae.loss_fn_lpips):
            non_vae_parameters += list(vae.loss_fn_lpips.parameters())
        if exists(vae.discr):
            non_vae_parameters += list(vae.discr.parameters())

        non_vae_param_set = set(non_vae_parameters)
        vae_parameters = [p for p in all_parameters if p not in non_vae_param_set]
        self.vae_parameters = vae_parameters

        self.optim = get_optimizer(vae_parameters, lr=lr, wd=wd)
        self.scheduler_optim = LinearWarmup_CosineAnnealing(
            optimizer=self.optim,
            linear_warmup_start_factor=linear_warmup_start_factor,
            linear_warmup_total_iters=linear_warmup_total_iters,
            cosine_annealing_T_max=cosine_annealing_T_max,
            cosine_annealing_eta_min=cosine_annealing_eta_min
        )
        
        self.scheduler_optim_overhead = scheduler_optim_overhead
        
        if exists(vae.discr):
            discr_parameters = list(vae.discr.parameters())
            self.discr_optim = get_optimizer(discr_parameters, lr=1e-6, wd=1e-4)
        else:
            self.discr_optim = None

        self.max_grad_norm = max_grad_norm
        self.discr_max_grad_norm = discr_max_grad_norm

        # dataloader

        self.dl = DataLoader(
            self.ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=32,
            pin_memory=True,
        )

        self.valid_dl = DataLoader(
            self.valid_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=32,
            pin_memory=True,
        )
        # prepare with accelerator

        prepare_args = [
            self.vae,
            self.optim,
            self.dl,
            self.valid_dl
        ]

        if exists(self.discr_optim):
            prepare_args.insert(2, self.discr_optim)  # Insert discr_optim after self.optim

        prepared = self.accelerator.prepare(*prepare_args)

        # Unpack depending on whether discriminator is included
        if exists(self.discr_optim):
            self.vae, self.optim, self.discr_optim, self.dl, self.valid_dl = prepared
        else:
            self.vae, self.optim, self.dl, self.valid_dl = prepared
        
        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.apply_grad_penalty_every = apply_grad_penalty_every

        self.results_folder = Path(results_folder)

        self.results_folder.mkdir(parents=True, exist_ok=True)

    def save(self, path):
        # Ensure the save path directory exists
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save the state dict of the model
        model_file = path / 'pytorch_model.bin'
        torch.save(self.vae.state_dict(), model_file)

        # If using EMA, save its state dict as well
        if self.use_ema:
            ema_model_file = path / 'ema_pytorch_model.bin'
            torch.save(self.ema_vae.ema_model.state_dict(), ema_model_file)

        # Save optimizer and other training states if needed
        optimizer_file = path / 'optimizer.bin'
        torch.save({
            'optimizer': self.optim.state_dict(),
            'steps': self.steps.item()
        }, optimizer_file)

        # Save config
        config_file = path / 'model_config.json'
        if hasattr(self.vae, 'get_config'):
            with open(config_file, 'w') as f:
                json.dump(self.vae.get_config(), f, indent=4)
                
    def load(self, path):
        path = Path(path)
        
        model_file = path / 'pytorch_model.bin'
        state_dict = torch.load(model_file, map_location=self.accelerator.device)

        if 'step' in state_dict:
            del state_dict['step']
        
        self.vae.load_state_dict(state_dict)
    
    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())
        apply_grad_penalty = not (steps % self.apply_grad_penalty_every)

        self.vae.train()
        logs = {}

        # update vae (generator)

        for _ in range(self.grad_accum_every):
            img = next(self.dl_iter)

            with self.accelerator.autocast():
                results = self.vae(img, current_step=steps)

            loss = results["total_loss"]

            self.accelerator.backward(loss / self.grad_accum_every)

            # Accumulate all logs
            for k, v in results.items():
                if k == "recon" or v is None:
                    continue
                accum_log(logs, {k: v.item() / self.grad_accum_every})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(
                self.vae.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        self.scheduler_optim.step(self.steps + self.scheduler_optim_overhead)
        accum_log(logs, {'lr': self.optim.param_groups[0]["lr"]})
        
        if exists(self.discr_optim) and self.steps.item() >= 0:
            self.discr_optim.zero_grad()

            for _ in range(self.grad_accum_every):
                img = next(self.dl_iter)

                loss = self.vae(img,
                                return_discr_loss=True,
                                apply_grad_penalty=apply_grad_penalty,
                                current_step=int(self.steps.item()))

                self.accelerator.backward(loss / self.grad_accum_every)

                accum_log(logs, {'discr_loss': loss.item() / self.grad_accum_every})

            if exists(self.discr_max_grad_norm):
                self.accelerator.clip_grad_norm_(
                    self.vae.discr.parameters(), self.discr_max_grad_norm)

            self.discr_optim.step()

        # update exponential moving averaged generator

        if self.is_main and self.use_ema:
            self.ema_vae.update()

        # sample results every so often

        if (self.steps == 0):
            self.valid_data_to_log = next(self.valid_dl_iter)[:4]

        if not (steps % self.save_results_every):
            vaes_to_evaluate = ((self.vae, str(steps)),)

            if self.use_ema:
                vaes_to_evaluate = (
                    (self.ema_vae.ema_model, f'{steps}.ema'),) + vaes_to_evaluate

            for model, filename in vaes_to_evaluate:
                model.eval()

                # valid_data = next(self.valid_dl_iter)

                is_video = self.valid_data_to_log.ndim == 5

                # valid_data = valid_data[:4].to(device)

                with torch.no_grad():
                    recons = model(self.valid_data_to_log, return_recons_only=True)

                # if is video, save gifs to folder
                # else save a grid of images

                if is_video:
                    sampled_videos_path = self.results_folder / \
                        f'samples.{filename}'
                    (sampled_videos_path).mkdir(parents=True, exist_ok=True)

                    for i, tensor in enumerate(recons.unbind(dim=0)):
                        video_tensor_to_gif(tensor.cpu(), str(
                            sampled_videos_path / f'{filename}-{i}.gif'))

                        if (i < 4):
                            self.accelerator.log({
                                f"image{i}": [wandb.Image(video_tensor_to_pil_first_image(tensor.cpu())),
                                              wandb.Image(video_tensor_to_pil_first_image(self.valid_data_to_log[i].cpu()))],
                            })


                else:
                    imgs_and_recons = torch.stack((self.valid_data_to_log, recons), dim=0)
                    imgs_and_recons = rearrange(
                        imgs_and_recons, 'r b ... -> (b r) ...')

                    imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0., 1.)
                    grid = make_grid(imgs_and_recons, nrow=2,
                                     normalize=True, range=(0, 1))

                    logs['reconstructions'] = grid

                    save_image(
                        grid, str(self.results_folder / f'{filename}.png'))

            self.print(f'{steps}: saving to {str(self.results_folder)}')

        # save model every so often

        if self.is_main and not (steps % self.save_model_every):

            save_path = Path(
                str(self.results_folder / f'ckpt_accelerate_{steps}/'))
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.save(save_path)

            self.print(f'{steps}: saving model to {str(save_path)}')

        self.steps += 1
        return logs

    def train(self, log_fn=noop):
        pbar = tqdm(total=self.num_train_steps,
                    initial=int(self.steps.item()),
                    disable=not self.is_main,
                    dynamic_ncols=True)

        while self.steps < self.num_train_steps:
            logs = self.train_step()

            if (
                self.accelerator.is_main_process
                and self.accelerator.trackers
                and self.steps.item() % self.log_every == 0
            ):
                log_dict = {}
                for k, v in logs.items():
                    log_dict[f"train/{k}"] = v
                self.accelerator.log(log_dict)

            # Update progress bar in terminal
            loss_display = logs.get("total_loss", 0.0)
            pbar.set_postfix(loss=f"{loss_display:.4f}")
            pbar.update(1)

            # Optional custom log handler (e.g., CSV logger or printing)
            log_fn(logs)

        pbar.close()
        self.print('training complete')
