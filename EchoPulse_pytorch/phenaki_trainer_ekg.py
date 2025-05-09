import math
import copy
from pathlib import Path
from random import random, choices
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

from beartype import beartype
from beartype.door import is_bearable
from beartype.vale import Is
from typing import Optional, List, Iterable, Tuple
from typing_extensions import Annotated

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset

from torch.optim import Adam

from torchvision import transforms as T
from torchvision.utils import make_grid, save_image

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm

from EchoPulse_pytorch.optimizer import get_optimizer, LinearWarmup_CosineAnnealing
from accelerate import Accelerator

# from phenaki_pytorch.phenaki_pytorch import Phenaki
from EchoPulse_pytorch.phenaki_pytorch_ekg import Phenaki
from EchoPulse_pytorch.data import MultiModalDatasetCMRECG, DataLoader, video_tensor_to_gif, video_tensor_to_pil_first_image

import os
import sys
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../..'))
sys.path.append(parent_dir)

from EchoPulse_pytorch.dataset_private_mp4_preprocess import EchoDataset_from_Video
# constants
from typing import Union
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np

def mask_schedule(ratio, total_unknown, method="cosine"):
    """Generates a mask rate by scheduling mask functions R.

    Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. During
    training, the input ratio is uniformly sampled; during inference, the input
    ratio is based on the step number divided by the total iteration number: t/T.
    Based on experiements, we find that masking more in training helps.
    
    Args:
        ratio: The uniformly sampled ratio [0, 1) as input.
        total_unknown: The total number of tokens that can be masked out. For
          example, in MaskGIT, total_unknown = 256 for 256x256 images and 1024 for
          512x512 images.
        method: implemented functions are ["uniform", "cosine", "pow", "log", "exp"]
          "pow2.5" represents x^2.5

    Returns:
        The mask rate (float).
    """
    if method == "uniform":
        mask_ratio = 1. - ratio
    elif "pow" in method:
        exponent = float(method.replace("pow", ""))
        mask_ratio = 1. - ratio**exponent
    elif method == "cosine":
        mask_ratio = np.cos(math.pi / 2. * ratio)
    elif method == "log":
        mask_ratio = -np.log2(ratio) / np.log2(total_unknown)
    elif method == "exp":
        mask_ratio = 1 - np.exp2(-np.log2(total_unknown) * (1 - ratio))
    
    # Clamps mask into [epsilon, 1)
    mask_ratio = np.clip(mask_ratio, 1e-6, 1.)
    return mask_ratio

DATASET_FIELD_TYPE_CONFIG = dict(
    videos = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim in {4, 5}]
    ],
    ekg = Union[List[str], torch.Tensor, None],  # 변경된 부분
    video_codebook_ids = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.long]
    ],
    video_frame_mask = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.bool]
    ],
    ekg_embeds = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim == 3]
    ],
)

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def elements_to_device_if_tensor(arr, device):
    output = []
    for el in arr:
        if isinstance(el, torch.Tensor):
            el = el.to(device)
        output.append(el)
    return output

def split_iterable(it, split_size):
    accum = []
    for ind in range(math.ceil(len(it) / split_size)):
        start_index = ind * split_size
        accum.append(it[start_index: (start_index + split_size)])
    return accum

def split(t, split_size = None):
    if not exists(split_size):
        return t

    if isinstance(t, torch.Tensor):
        return t.split(split_size, dim = 0)

    if isinstance(t, Iterable):
        return split_iterable(t, split_size)

    return TypeError

def find_first(cond, arr):
    for el in arr:
        if cond(el):
            return el
    return None

def split_args_and_kwargs(*args, batch_size = None, split_size = None, **kwargs):
    all_args = (*args, *kwargs.values())
    len_all_args = len(all_args)

    if not exists(batch_size):
        first_tensor = find_first(lambda t: isinstance(t, torch.Tensor), all_args)
        assert exists(first_tensor)
        batch_size = len(first_tensor)

    split_size = default(split_size, batch_size)
    num_chunks = math.ceil(batch_size / split_size)

    dict_len = len(kwargs)
    dict_keys = kwargs.keys()
    split_kwargs_index = len_all_args - dict_len

    split_all_args = [split(arg, split_size = split_size) if exists(arg) and isinstance(arg, (torch.Tensor, Iterable)) else ((arg,) * num_chunks) for arg in all_args]
    chunk_sizes = tuple(map(len, split_all_args[0]))

    for (chunk_size, *chunked_all_args) in tuple(zip(chunk_sizes, *split_all_args)):
        chunked_args, chunked_kwargs_values = chunked_all_args[:split_kwargs_index], chunked_all_args[split_kwargs_index:]
        chunked_kwargs = dict(tuple(zip(dict_keys, chunked_kwargs_values)))
        chunk_size_frac = chunk_size / batch_size
        yield chunk_size_frac, (chunked_args, chunked_kwargs)

def simple_slugify(text, max_length = 255):
    return text.replace('-', '_').replace(',', '').replace(' ', '_').replace('|', '--').strip('-_')[:max_length]

def has_duplicates(tup):
    counts = dict()
    for el in tup:
        if el not in counts:
            counts[el] = 0
        counts[el] += 1
    return any(filter(lambda count: count > 1, counts.values()))

def determine_types(data, config):
    output = []
    for el in data:
        for name, data_type in config.items():
            if is_bearable(el, data_type):
                output.append(name)
                break
        else:
            raise TypeError(f'unable to determine type of {data}')

    return tuple(output)


# trainer class

@beartype
class PhenakiTrainer(object):
    def __init__(
        self,
        phenaki: Phenaki,
        *,
        folder = None,
        train_on_images = False,
        batch_size = 16,
        grad_accum_every = 1,
        num_frames = 17,
        sample_num_frames = None,
        train_lr = 1e-4,
        train_num_steps = 100000,
        max_grad_norm = None,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        wd = 0,
        linear_warmup_start_factor=0.1,
        linear_warmup_total_iters=100,
        cosine_annealing_T_max=1000000,
        cosine_annealing_eta_min=1e-5,
        save_and_sample_every = 1000,
        logging_steps = 1000,
        num_samples = 25,
        results_folder = './results',
        fp16 = False,
        split_batches = True,
        convert_image_to = None,
        sample_texts_file_path = None,  # path to a text file with video captions, delimited by newline
        sample_texts: Optional[List[str]] = None,
        dataset: Optional[Dataset] = None,
        dataset_fields: Optional[Tuple[str, ...]] = None,
        wandb_mode: str = 'disabled',
    ):
        
        super().__init__()
        maskgit = phenaki.maskgit
        cvivit = phenaki.cvivit
        self.wandb_mode = wandb_mode
        
        self.logging_steps = logging_steps
        assert exists(cvivit), 'cvivit must be present on phenaki'

        # define accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no',
            device_placement=True,
            log_with='wandb',
        )
        
        config = {
            key: value
            for key, value in locals().items()
            if key not in {'self', 'config', '__class__', 'cvivit', 'phenaki', 'maskgit'} and not key.startswith('__')
        }
        
        self.accelerator.init_trackers(
            project_name="CViViT",
            config=config,
            init_kwargs={"wandb": {"mode": self.wandb_mode}}
        )

        self.model = phenaki

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.unconditional = maskgit.unconditional

        # training related variables

        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps
        self.image_size = cvivit.image_size

        # sampling related variables

        self.num_samples = num_samples

        self.sample_texts = None

        if exists(sample_texts_file_path):   ##在这里读取caption
            sample_texts_file_path = Path(sample_texts_file_path)
            # print(sample_texts_file_path)
            assert sample_texts_file_path.exists()
            captions = sample_texts_file_path.read_text().split('\n')
            print(len(captions))
            # self.sample_texts = list(filter(len, captions)) ## To-Dos
            self.sample_texts = list(captions)
        
        elif exists(self.sample_texts):
            self.sample_texts = sample_texts
            
        print("---------------------------------------------------------------------------------------------------------------------------------")
        # print(len(self.sample_texts))
        # assert maskgit.unconditional or exists(self.sample_texts), 'if maskgit is to be trained text conditioned, `sample_texts` List[str] or `sample_texts_file_path` must be given'

        self.save_and_sample_every = save_and_sample_every

        # User specific Dataset
        self.sample_num_frames = default(sample_num_frames, num_frames)
        self.train_on_images = train_on_images

        self.ds = MultiModalDatasetCMRECG(
            mode='train',
            image_size=self.image_size,
            folder_cmr=f'{folder}/imaging_pretrain/',
            folder_ecg=f'{folder}/multimodal/ecg_ED_ES/',
        )

        # self.sampler = DistributedSampler(self.ds, num_replicas=self.world_size, rank=self.rank, shuffle=True)

        dl = DataLoader(
            self.ds,
            batch_size = batch_size,
            pin_memory = True,
            num_workers = 32
            )

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        if exists(dataset_fields):
            assert not has_duplicates(dataset_fields), 'dataset fields must not have duplicate field names'
            valid_dataset_fields = set(DATASET_FIELD_TYPE_CONFIG.keys())
            assert len(set(dataset_fields) - valid_dataset_fields) == 0, f'dataset fields must be one of {valid_dataset_fields}'

        self.dataset_fields = dataset_fields

        # optimizer

        self.opt = get_optimizer(maskgit.parameters(), lr = train_lr, wd = wd, betas = adam_betas)
        
        self.scheduler_optim = LinearWarmup_CosineAnnealing(
            optimizer=self.opt,
            linear_warmup_start_factor=linear_warmup_start_factor,
            linear_warmup_total_iters=linear_warmup_total_iters,
            cosine_annealing_T_max=cosine_annealing_T_max,
            cosine_annealing_eta_min=cosine_annealing_eta_min
        )

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents = True, exist_ok = True)

    def wrap_distributed(self):
        self.model = DDP(self.model, device_ids=[self.rank])
        
    def data_tuple_to_kwargs(self, data):
        if not exists(self.dataset_fields):
            self.dataset_fields = determine_types(data, DATASET_FIELD_TYPE_CONFIG)
            assert not has_duplicates(self.dataset_fields), 'dataset fields must not have duplicate field names'

        return dict(zip(self.dataset_fields, data))

        # fields = self.DATASET_FIELD_TYPE_CONFIG.keys()
        
        # if len(data) > len(fields):
        #     # EKG 데이터가 추가되었다고 가정
        #     data_dict = dict(zip(fields, data[:len(fields)]))
        #     data_dict['ekg'] = data[len(fields)]  # EKG 데이터 추가
        # else:
        #     data_dict = dict(zip(fields, data))

        # return data_dict

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

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train_step(
        self,
        only_train_generator = False,
        only_train_critic = False
    ):
        accelerator = self.accelerator
        device = self.device

        total_loss = 0.

        for _ in range(self.grad_accum_every):
            data = next(self.dl)
            # data = elements_to_device_if_tensor(data, device)
            data_kwargs = self.data_tuple_to_kwargs(data)

            assert not (self.train_on_images and data_kwargs['videos'].ndim != 4), 'you have it set to train on images, but the dataset is not returning tensors of 4 dimensions (batch, channels, height, width)'

            with self.accelerator.autocast():
                loss = self.model(**{
                    **data_kwargs,
                    'only_train_generator': only_train_generator,
                    'only_train_critic': only_train_critic
                })
                loss = loss / self.grad_accum_every
                if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                    loss = loss.mean()
                total_loss += loss.item()

            self.accelerator.backward(loss)

        if exists(self.max_grad_norm):
            accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        accelerator.wait_for_everyone()

        self.opt.step()
        self.opt.zero_grad()
        
        self.scheduler_optim.step(self.step)

        accelerator.wait_for_everyone()

        if self.is_main and self.step % self.save_and_sample_every == 0:
            milestone = self.step // self.save_and_sample_every
            self.save(milestone)
            self.evaluate_and_visualize(num_eval_samples=self.num_samples)
            
        self.step += 1
        
        return total_loss
    
    def evaluate_and_visualize(self, num_eval_samples: int = 4):
        self.model.eval()

        batch = next(self.dl)
        # batch = elements_to_device_if_tensor(batch, self.device)
        data_kwargs = self.data_tuple_to_kwargs(batch)

        ekgs = data_kwargs['ekg'][:num_eval_samples]

        with torch.no_grad():
            generated_videos = self.model.sample(
                ekg=ekgs,
                num_frames=self.sample_num_frames,
                batch_size=ekgs.size(0),
            )  # [B, C, T, H, W]

        out_folder = self.results_folder / f'eval_step_{self.step}'
        out_folder.mkdir(parents=True, exist_ok=True)

        for i, video_tensor in enumerate(generated_videos):
            gif_path = out_folder / f"video_{i}.gif"
            video_tensor_to_gif(video_tensor.cpu(), gif_path)

            # Log to WandB if enabled
            if self.accelerator.is_main_process and self.accelerator.trackers:
                import wandb
                self.accelerator.log({
                    f"generated/video_{i}": wandb.Image(video_tensor_to_pil_first_image(video_tensor.cpu()))
                })

        self.model.train()

    def train(
        self,
        only_train_generator=False,
        only_train_critic=False
    ):

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not self.is_main,
            dynamic_ncols=True,
        ) as pbar:
            
            step_losses = []

            while self.step < self.train_num_steps:

                self.model.training_mask_ratio = float(self.step)/float(self.train_num_steps)
                
                loss = self.train_step(
                    only_train_generator=only_train_generator,
                    only_train_critic=only_train_critic
                )

                step_losses.append(loss)
                pbar.set_postfix(loss=f"{loss:.4f}", refresh=False)
                pbar.update(1)

                # Log average loss every 1000 steps
                if self.step % self.logging_steps == 0 and self.step > 0:
                    if self.accelerator.is_main_process and self.accelerator.trackers:
                        avg_loss = sum(step_losses) / len(step_losses)
                        self.accelerator.log({"train/loss": avg_loss})
                    step_losses = []

        self.print('training complete')

