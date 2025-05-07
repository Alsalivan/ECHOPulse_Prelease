import torch
from EchoPulse_pytorch.cvivit import CViViT
from EchoPulse_pytorch.cvivit_trainer import CViViTTrainer
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs


# Define the configuration for Accelerator
accelerate_kwargs = {
    'mixed_precision': 'fp16',  # use mixed precision training
    'split_batches': True,
    'log_with': 'wandb',
}

wandb_mode = 'online'

# Initialize the CViViT model
cvivit = CViViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 84,
    patch_size = 7,
    channels = 1,
    temporal_patch_size = 2,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 64,
    heads = 8,
    use_discr=False,
    attn_dropout=0.1,
    ff_dropout=0.1,
    commit_loss_w=1.0,
    gen_loss_w=0.0,
    perceptual_loss_w=0.0,
    i3d_loss_w=0.0,
    recon_loss_w=1.0,
    wandb_mode=wandb_mode,
    lookup_free_quantization=False,
)

# Use Accelerator for model and data preparation
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(**accelerate_kwargs, kwargs_handlers=[kwargs])
cvivit = cvivit.to(accelerator.device)

import random
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
randint = random.randint(0, 100000)
results_folder = f'./results/cvivit/train_{randint}_{timestamp}/'

# Initialize the trainer
trainer = CViViTTrainer(
    vae=cvivit,  # Pass the unwrapped model
    accelerator=accelerator,
    folder='/vol/miltank/users/seliv/Documents/projects/output/ecgcmr/saved_tensors/imaging_pretrain',
    batch_size=64,
    num_frames=25,
    grad_accum_every=4,
    train_on_images=False,
    wd=1e-7,
    use_ema=False,
    num_train_steps=120000,
    save_model_every=5000,
    save_results_every=1000,
    cosine_annealing_T_max=120000,
    log_every=100,
    wandb_mode=wandb_mode,
    results_folder=results_folder,
    accelerate_kwargs=accelerate_kwargs,
    discr_max_grad_norm=1.0,
)

# Start training
trainer.train()  # Reconstructions and checkpoints will be saved periodically to ./results
