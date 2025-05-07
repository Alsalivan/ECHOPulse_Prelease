from pathlib import Path
import copy
import math
from functools import wraps
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from torch import nn, einsum
from torch.autograd import grad as torch_grad

import torchvision

from torch.autograd import Variable
from EchoPulse_pytorch.frozen_models.pytorch_i3d.pytorch_i3d import InceptionI3d

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from vector_quantize_pytorch import VectorQuantize, LFQ

from EchoPulse_pytorch.attention import Attention, Transformer, ContinuousPositionBias
from safetensors.torch import load_file  # 需要安装 safetensors 库
import lpips

def preprocess_for_lpips(x):
    return x.clamp(0., 1.) * 2. - 1.

log_formats = False

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def divisible_by(numer, denom):
    return (numer % denom) == 0


def leaky_relu(p=0.1):
    return nn.LeakyReLU(p)


def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, 'vgg')
        if has_vgg:
            vgg = self.vgg
            delattr(self, 'vgg')

        out = fn(self, *args, **kwargs)

        if has_vgg:
            self.vgg = vgg

        return out
    return inner


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


def cast_tuple(val, l=1):
    return val if isinstance(val, tuple) else (val,) * l


def gradient_penalty(images, output, weight=10):
    batch_size = images.shape[0]

    gradients = torch_grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=images.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def l2norm(t):
    return F.normalize(t, dim=-1)


def leaky_relu(p=0.1):
    return nn.LeakyReLU(p)


def safe_div(numer, denom, eps=1e-8):
    return numer / (denom + eps)

# gan losses


def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()


def hinge_gen_loss(fake):
    return -fake.mean()


def bce_discr_loss(fake, real):
    return (-F.log(1 - torch.sigmoid(fake)) - F.log(torch.sigmoid(real))).mean()


def bce_gen_loss(fake):
    return -F.log(torch.sigmoid(fake)).mean()


def grad_layer_wrt_loss(loss, layer):
    return torch_grad(
        outputs=loss,
        inputs=layer,
        grad_outputs=torch.ones_like(loss),
        retain_graph=True
    )[0].detach()


def log_wandb_all_losses(accelerator, commit_loss, gen_loss, perceptual_loss, i3d_video_perceptual_loss, recon_loss):
    accelerator.log({"commit_loss": commit_loss.item()})
    accelerator.log({"gen_loss": gen_loss.item()})
    accelerator.log({"perceptual_loss": perceptual_loss.item()})
    accelerator.log({"i3d_video_perceptual_loss": i3d_video_perceptual_loss.item()})
    accelerator.log({"recon_loss": recon_loss.item()})

    return

# discriminator

def center_crop_64(x):
    _, _, h, w = x.shape
    start_h = (h - 64) // 2
    start_w = (w - 64) // 2
    return x[:, :, start_h:start_h + 64, start_w:start_w + 64]


class DiscriminatorBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        filters,
        downsample=True
    ):
        super().__init__()
        self.conv_res = nn.Conv2d(
            input_channels, filters, 1, stride=(2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
            nn.Conv2d(filters * 4, filters, 1)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)

        if exists(self.downsample):
            x = self.downsample(x)

        x = (x + res) * (1 / math.sqrt(2))
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        channels=3,
        attn_res_layers=(16,),
        max_dim=512
    ):
        super().__init__()
        image_size = pair(image_size)
        min_image_resolution = min(image_size)

        num_layers = int(math.log2(min_image_resolution) - 2)
        attn_res_layers = cast_tuple(attn_res_layers, num_layers)

        blocks = []

        layer_dims = [channels] + [(dim * 4) * (2 ** i)
                                   for i in range(num_layers + 1)]
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))

        blocks = []
        attn_blocks = []

        image_resolution = min_image_resolution

        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(layer_dims_in_out) - 1)

            block = DiscriminatorBlock(
                in_chan, out_chan, downsample=is_not_last)
            blocks.append(block)

            attn_block = None
            if image_resolution in attn_res_layers:
                attn_block = Attention(dim=out_chan)

            attn_blocks.append(attn_block)

            image_resolution //= 2

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)

        dim_last = layer_dims[-1]

        downsample_factor = 2 ** num_layers
        last_fmap_size = tuple(
            map(lambda n: n // downsample_factor, image_size))

        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last

        self.to_logits = nn.Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding=1),
            leaky_relu(),
            Rearrange('b ... -> b (...)'),
            nn.Linear(latent_dim, 1),
            Rearrange('b 1 -> b')
        )

    def forward(self, x):

        for block, attn_block in zip(self.blocks, self.attn_blocks):
            x = block(x)

            if exists(attn_block):
                x, ps = pack([x], 'b c *')
                x = rearrange(x, 'b c n -> b n c')
                x = attn_block(x) + x
                x = rearrange(x, 'b n c -> b c n')
                x, = unpack(x, ps, 'b c *')

        return self.to_logits(x)

# c-vivit - 3d ViT with factorized spatial and temporal attention made into an vqgan-vae autoencoder


def pick_video_frame(video, frame_indices):
    batch, device = video.shape[0], video.device
    video = rearrange(video, 'b c f ... -> b f c ...')
    batch_indices = torch.arange(batch, device=device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')
    images = video[batch_indices, frame_indices]
    images = rearrange(images, 'b 1 c ... -> b c ...')
    return images

def ensure_three_channels(x):
    if x.shape[1] == 1:
        return x.repeat(1, 3, *([1] * (x.ndim - 2)))
    return x


@torch.no_grad()
def i3d_inference(video, i3d_model):
    """
    Inputs:
        video: Tensor of shape [B, C, T, H, W] in range [0, 1]
        i3d_model: pretrained, frozen InceptionI3D model

    Returns:
        features: Tensor of shape [B, C]
    """

    b, c, t, h, w = video.shape

    # Convert grayscale to RGB if needed
    if c == 1:
        video = video.repeat(1, 3, 1, 1, 1)

    # Normalize to [-1, 1] (I3D expects this)
    video = video.clamp(0., 1.) * 2. - 1.

    # Resize spatial dimensions to 224x224
    video = rearrange(video, 'b c t h w -> (b t) c h w')
    video = T.Resize((224, 224))(video)
    video = rearrange(video, '(b t) c h w -> b c t h w', b=b, t=t)

    # Extract features from I3D
    features = i3d_model.extract_features(video)  # [B, C, T', H', W']

    # Global average pool over T, H, W → shape: [B, C, 1, 1, 1]
    features = F.adaptive_avg_pool3d(features, 1)

    # Flatten to [B, C]
    return features.view(b, -1)

class CViViT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        codebook_size,
        image_size,
        patch_size,
        temporal_patch_size,
        spatial_depth,
        temporal_depth,
        wandb_mode="disabled",
        codebook_dim=32,
        discr_base_dim=16,
        dim_head=64,
        heads=8,
        channels=3,
        force_cpu=False,
        i3d_model_path='EchoPulse_pytorch/frozen_models/pytorch_i3d/models/rgb_imagenet.pt',
        discr_attn_res_layers=(16,),
        use_hinge_loss=True,
        attn_dropout=0.,
        ff_dropout=0.,
        ff_mult=4.,
        lookup_free_quantization = True,
        lookup_free_quantization_kwargs: dict = {},

        commit_loss_w=1.,
        commitment_weight=0.25,
        gen_loss_w=1.,
        perceptual_loss_w=1.,
        i3d_loss_w=1.,
        recon_loss_w=1.,

        use_discr=True,
        gp_weight=10000
    ):
        """
        einstein notations:

        b - batch
        c - channels
        t - time
        d - feature dimension
        p1, p2, pt - image patch sizes and then temporal patch size
        """

        super().__init__()
        
        arguments = locals()
        self.config = {}
        for key in arguments:
            if key not in ['self', '__class__']:
                self.config[key] = arguments[key]

        self.wandb_mode = wandb_mode
        self.force_cpu = force_cpu

        self.commit_loss_w = commit_loss_w
        self.gen_loss_w = gen_loss_w
        self.perceptual_loss_w = perceptual_loss_w
        self.i3d_loss_w = i3d_loss_w
        self.recon_loss_w = recon_loss_w

        self.gp_weight = gp_weight

        self.use_discr = use_discr

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        self.temporal_patch_size = temporal_patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(
            dim=dim, heads=heads)

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (
            image_width % patch_width) == 0

        self.to_patch_emb_first_frame = nn.Sequential(
            Rearrange('b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)',
                      p1=patch_height, p2=patch_width),
            nn.LayerNorm(channels * patch_width * patch_height),
            nn.Linear(channels * patch_width * patch_height, dim),
            nn.LayerNorm(dim)
        )

        self.to_patch_emb = nn.Sequential(
            Rearrange('b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)',
                      p1=patch_height, p2=patch_width, pt=temporal_patch_size),
            nn.LayerNorm(channels * patch_width *
                         patch_height * temporal_patch_size),
            nn.Linear(channels * patch_width *
                      patch_height * temporal_patch_size, dim),
            nn.LayerNorm(dim)
        )

        spatial_transformer_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            causal = False,
            peg = False,
            ff_mult = ff_mult,
        )

        # only temporal transformers have PEG and are causal

        temporal_transformer_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            causal = True,
            peg = True,
            peg_causal = True,
            ff_mult = ff_mult,
        )

        self.enc_spatial_transformer = Transformer(depth = spatial_depth, **spatial_transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth = temporal_depth, **temporal_transformer_kwargs)

        # offer look up free quantization
        # https://arxiv.org/abs/2310.05737

        self.lookup_free_quantization = lookup_free_quantization

        if lookup_free_quantization:
            self.vq = LFQ(
                dim = dim,
                codebook_size = codebook_size,
                experimental_softplus_entropy_loss=True,
                **lookup_free_quantization_kwargs)
        else:
            self.vq = VectorQuantize(
                dim = dim,
                codebook_dim = codebook_dim,
                codebook_size = codebook_size,
                commitment_weight = commitment_weight,
                use_cosine_sim = True)
            
        self.dec_spatial_transformer = Transformer(depth = spatial_depth, **spatial_transformer_kwargs)
        self.dec_temporal_transformer = Transformer(depth = temporal_depth, **temporal_transformer_kwargs)

        self.to_pixels_first_frame = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height),
            Rearrange('b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)',
                      p1=patch_height, p2=patch_width)
        )

        self.to_pixels = nn.Sequential(
            nn.Linear(dim, channels * patch_width *
                      patch_height * temporal_patch_size),
            Rearrange('b t h w (c pt p1 p2) -> b c (t pt) (h p1) (w p2)',
                      p1=patch_height, p2=patch_width, pt=temporal_patch_size),
        )

        # === Optional LPIPS Perceptual Loss ===
        if self.perceptual_loss_w > 0:
            self.loss_fn_lpips = lpips.LPIPS(net='vgg').requires_grad_(False)
            if not self.force_cpu:
                self.loss_fn_lpips.cuda()
            else:
                self.loss_fn_lpips.cpu()
            self.loss_fn_lpips.eval()
        else:
            self.loss_fn_lpips = None

        # === Optional I3D Video Perceptual Loss ===
        if self.i3d_loss_w > 0:
            self.i3d = InceptionI3d(400, in_channels=3)
            self.i3d.load_state_dict(torch.load(i3d_model_path))
            for param in self.i3d.parameters():
                param.requires_grad = False
            self.i3d.eval()
        else:
            self.i3d = None

        # === Optional Discriminator (GAN) Losses ===
        if self.gen_loss_w > 0 and use_discr:
            self.discr = Discriminator(
                image_size=(64, 64),
                dim=discr_base_dim,
                channels=channels,
                attn_res_layers=discr_attn_res_layers
            )
            self.discr_loss = hinge_discr_loss if use_hinge_loss else bce_discr_loss
            self.gen_loss = hinge_gen_loss if use_hinge_loss else bce_gen_loss
        else:
            self.discr = None
            self.discr_loss = None
            self.gen_loss = None
    
    def get_config(self):
        return self.config

    def calculate_video_token_mask(self, videos, video_frame_mask):
        *_, h, w = videos.shape
        ph, pw = self.patch_size

        assert torch.all(((video_frame_mask.sum(dim=-1) - 1) % self.temporal_patch_size) ==
                         0), 'number of frames must be divisible by temporal patch size, subtracting off the first frame'
        first_frame_mask, rest_frame_mask = video_frame_mask[:,
                                                             :1], video_frame_mask[:, 1:]
        rest_vq_mask = rearrange(
            rest_frame_mask, 'b (f p) -> b f p', p=self.temporal_patch_size)
        video_mask = torch.cat(
            (first_frame_mask, rest_vq_mask.any(dim=-1)), dim=-1)
        print(video_mask)
        return repeat(video_mask, 'b f -> b (f hw)', hw=(h // ph) * (w // pw))
    
    def calculate_video_token_mask_firstframe(self, videos, video_frame_mask):
        *_, h, w = videos.shape
        ph, pw = self.patch_size

        # 确保其余帧的数量能够整除时间补丁大小
        assert torch.all(((video_frame_mask.sum(dim=-1) - 1) % self.temporal_patch_size) ==
                        0), 'number of frames must be divisible by temporal patch size, subtracting off the first frame'

        # 只处理其余的帧遮罩
        rest_frame_mask = video_frame_mask[:, 1:]

        # 对其余帧进行重排
        rest_vq_mask = rearrange(
            rest_frame_mask, 'b (f p) -> b f p', p=self.temporal_patch_size)

        # 不包括第一帧的遮罩，只对其余帧取逻辑或
        video_mask = rest_vq_mask.any(dim=-1)

        # 重复遮罩以适应空间的Token
        return repeat(video_mask, 'b f -> b (f hw)', hw=(h // ph) * (w // pw))
    
    def get_video_patch_shape(self, num_frames, include_first_frame=True):
        patch_frames = 0

        if include_first_frame:
            num_frames -= 1
            patch_frames += 1

        patch_frames += (num_frames // self.temporal_patch_size)

        return (patch_frames, *self.patch_height_width)

    @property
    def image_num_tokens(self):
        return int(self.image_size[0] / self.patch_size[0]) * int(self.image_size[1] / self.patch_size[1])

    def frames_per_num_tokens(self, num_tokens):
        tokens_per_frame = self.image_num_tokens

        assert (num_tokens %
                tokens_per_frame) == 0, f'number of tokens must be divisible by number of tokens per frame {tokens_per_frame}'
        assert (num_tokens > 0)

        # HUGO CHECK
        pseudo_frames = num_tokens // tokens_per_frame
        return (pseudo_frames - 1) * self.temporal_patch_size + 1

    def num_tokens_per_frames(self, num_frames, include_first_frame=True):
        image_num_tokens = self.image_num_tokens

        total_tokens = 0

        if include_first_frame:
            num_frames -= 1
            total_tokens += image_num_tokens

        assert (num_frames % self.temporal_patch_size) == 0

        return total_tokens + int(num_frames / self.temporal_patch_size) * image_num_tokens
    
    def copy_for_eval(self):
        device = next(self.parameters()).device
        vae_copy = copy.deepcopy(self.cpu())

        # Remove optional modules if they were initialized
        if hasattr(vae_copy, 'discr'):
            del vae_copy.discr
        if hasattr(vae_copy, 'loss_fn_lpips'):
            del vae_copy.loss_fn_lpips
        if hasattr(vae_copy, 'i3d'):
            del vae_copy.i3d

        vae_copy.eval()
        return vae_copy.to(device)

    @remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    @remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path), map_location='cpu')
        self.load_state_dict(pt)
    
    def load_safetensors(self, path):
        path = Path(path)
        assert path.exists(), "路径不存在"
        
        if path.suffix == '.bin':
            pt = torch.load(str(path))
        elif path.suffix == '.safetensors':
            pt = load_file(str(path))
        else:
            raise ValueError("不支持的文件格式")
        # Print keys in the checkpoint file
        print("Keys in checkpoint file:", pt.keys())

        # Print keys in the model's state dictionary
        print("Keys in model's state dictionary:", self.state_dict().keys())

        # Load the state dictionary with strict=False
        missing_keys, unexpected_keys = self.load_state_dict(pt, strict=False)
        self.load_state_dict(pt)
    
    
    def decode_from_codebook_indices(self, indices):
        codes = self.vq.codebook[indices]
        projected_out_codes = self.vq.project_out(codes)
        return self.decode(projected_out_codes)

    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def encode(
        self,
        tokens
    ):
        b = tokens.shape[0]  # batch size
        h, w = self.patch_height_width  # patch h,w

        # video shape, last dimension is the embedding size
        video_shape = tuple(tokens.shape[:-1])

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)
        # encode - spatial

        tokens = self.enc_spatial_transformer(
            tokens, attn_bias=attn_bias, video_shape=video_shape)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b, h=h, w=w)

        # encode - temporal

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        tokens = self.enc_temporal_transformer(tokens, video_shape=video_shape)

        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b=b, h=h, w=w)

        return tokens

    def decode(
        self,
        tokens
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        if tokens.ndim == 3:
            tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h=h, w=w)

        video_shape = tuple(tokens.shape[:-1])

        # decode - temporal

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        tokens = self.dec_temporal_transformer(tokens, video_shape=video_shape)

        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b=b, h=h, w=w)

        # decode - spatial

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)

        tokens = self.dec_spatial_transformer(
            tokens, attn_bias=attn_bias, video_shape=video_shape)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b, h=h, w=w)

        # to pixels

        first_frame_token, rest_frames_tokens = tokens[:, :1], tokens[:, 1:]

        first_frame = self.to_pixels_first_frame(first_frame_token)

        rest_frames = self.to_pixels(rest_frames_tokens)

        recon_video = torch.cat((first_frame, rest_frames), dim=2)

        return recon_video

    def forward(
        self,
        video,
        mask=None,
        return_recons=False,
        return_recons_only=False,
        return_discr_loss=False,
        apply_grad_penalty=True,
        return_only_codebook_ids=False,
        current_step: Optional[int] = None,
    ):

        # FORWARD PASS

        # 4 is BxCxHxW (for images), 5 is BxCxFxHxW
        assert video.ndim in {4, 5}

        is_image = video.ndim == 4

        if is_image:  # add temporal channel to 1 for images only
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert not exists(mask)

        b, c, f, *image_dims, device = *video.shape, video.device
        
        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f
        assert divisible_by(
            f - 1, self.temporal_patch_size), f'number of frames ({f}) minus one ({f - 1}) must be divisible by temporal patch size ({self.temporal_patch_size})'

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]

        # derive patches

        first_frame_tokens = self.to_patch_emb_first_frame(first_frame)
        rest_frames_tokens = self.to_patch_emb(rest_frames)

        # simple cat, normal
        tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim=1)

        # save height and width in

        shape = tokens.shape
        *_, h, w, _ = shape

        # encode - spatial

        tokens = self.encode(tokens)

        # quantize

        tokens, packed_fhw_shape = pack([tokens], 'b * d')

        vq_mask = None
        if exists(mask):
            vq_mask = self.calculate_video_token_mask(video, mask)
            
        vq_kwargs = dict(mask = vq_mask) if not self.lookup_free_quantization else dict()

        tokens, indices, commit_loss = self.vq(tokens, **vq_kwargs)

        if return_only_codebook_ids:
            indices, = unpack(indices, packed_fhw_shape, 'b *')
            return indices

        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h=h, w=w)

        recon_video = self.decode(tokens)

        # handle shape if we are training on images only
        returned_recon = rearrange(
            recon_video, 'b c 1 h w -> b c h w') if is_image else recon_video.clone()

        if return_recons_only:
            return returned_recon

        # LOSS COMPUTATION
        if exists(mask):
            # variable lengthed video / images training
            recon_loss = F.mse_loss(video, recon_video, reduction = 'none')
            recon_loss = recon_loss[repeat(mask, 'b t -> b c t', c = c)]
            recon_loss = recon_loss.mean()
        else:
            recon_loss = F.mse_loss(video, recon_video)
            
            
        # === Early return if no other losses are used ===
        if self.perceptual_loss_w == 0 and self.i3d_loss_w == 0 and self.gen_loss_w == 0:
            loss = self.commit_loss_w * commit_loss + self.recon_loss_w * recon_loss
            if return_recons:
                return loss, returned_recon
            return {
                "total_loss": loss,
                "commit_loss": commit_loss,
                "recon_loss": recon_loss,
            }

        # === Pick random frame only if needed ===
        needs_frame = (self.perceptual_loss_w > 0 or self.gen_loss_w > 0 or return_discr_loss)
        if needs_frame:
            pick_frame_logits = torch.randn(b, f, device=video.device)
            if exists(mask):
                mask_value = -torch.finfo(pick_frame_logits.dtype).max
                pick_frame_logits = pick_frame_logits.masked_fill(~mask, mask_value)
                
            frame_indices = pick_frame_logits.topk(1, dim=-1).indices
            video_frame = pick_video_frame(video, frame_indices)
            recon_frame = pick_video_frame(recon_video, frame_indices)
    
        # whether to return discriminator loss
        if return_discr_loss:
            assert exists(self.discr), 'discriminator must exist to train it'
            
            video_frame_cropped = center_crop_64(video_frame).requires_grad_()
            recon_frame_cropped = center_crop_64(recon_frame).detach()

            recon_video_discr_logits, video_discr_logits = map(self.discr, (recon_frame_cropped, video_frame_cropped))
            
            if torch.isinf(recon_video_discr_logits).any() or torch.isnan(recon_video_discr_logits).any():
                print("⚠️  Discriminator path - fake logits:", recon_video_discr_logits.min().item(), recon_video_discr_logits.max().item())

            if torch.isinf(video_discr_logits).any() or torch.isnan(video_discr_logits).any():
                print("⚠️  Discriminator path - real logits:", video_discr_logits.min().item(), video_discr_logits.max().item())

            discr_loss = self.discr_loss(recon_video_discr_logits, video_discr_logits)

            if apply_grad_penalty:
                gp = gradient_penalty(video_frame_cropped, video_discr_logits)
                loss = discr_loss + gp
            else:
                loss = discr_loss

            if return_recons:
                return loss, returned_recon

            return loss

        # === LPIPS perceptual loss ===
        if self.perceptual_loss_w > 0 and self.loss_fn_lpips is not None:
            if video_frame.shape[1] == 1:
                video_frame_vgg = video_frame.repeat(1, 3, 1, 1)
                recon_frame_vgg = recon_frame.repeat(1, 3, 1, 1)
            else:
                video_frame_vgg = video_frame
                recon_frame_vgg = recon_frame

            video_frame_vgg, recon_frame_vgg = map(preprocess_for_lpips, (video_frame_vgg, recon_frame_vgg))

            perceptual_loss = self.loss_fn_lpips(video_frame_vgg, recon_frame_vgg).mean()
        else:
            perceptual_loss = torch.tensor(0., device=video.device)

        # === I3D perceptual loss ===
        if self.i3d_loss_w > 0 and self.i3d is not None and video.shape[2] > 1:
            features_video = i3d_inference(video, self.i3d)
            features_recon_video = i3d_inference(recon_video, self.i3d)
            i3d_video_perceptual_loss = F.mse_loss(features_video, features_recon_video)
        else:
            i3d_video_perceptual_loss = torch.tensor(0., device=video.device)

        # === Generator (GAN) loss ===
        if self.gen_loss_w > 0 and self.discr is not None and current_step is not None and current_step > 30000:
            discr_input = center_crop_64(recon_frame)
            fake_logits = self.discr(discr_input)
            
            if torch.isinf(fake_logits).any() or torch.isnan(fake_logits).any():
                print("⚠️  Generator path - bad logits:", fake_logits.min().item(), fake_logits.max().item())

            gen_loss = self.gen_loss(fake_logits)
        else:
            gen_loss = torch.tensor(0., device=video.device)

        # calculate adaptive weight

        # last_dec_layer = self.to_pixels[0].weight

        # norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p = 2)
        # norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p = 2)

        # adaptive_weight = safe_div(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
        # adaptive_weight.clamp_(max = 1e4)

        loss = (
            self.commit_loss_w * commit_loss +
            self.gen_loss_w * gen_loss +
            self.perceptual_loss_w * perceptual_loss +
            self.i3d_loss_w * i3d_video_perceptual_loss +
            self.recon_loss_w * recon_loss
        )

        if return_recons:
            return loss, returned_recon

        return {
            "total_loss": loss,
            "commit_loss": commit_loss,
            "gen_loss": gen_loss,
            "perceptual_loss": perceptual_loss,
            "i3d_video_perceptual_loss": i3d_video_perceptual_loss,
            "recon_loss": recon_loss,
        }
