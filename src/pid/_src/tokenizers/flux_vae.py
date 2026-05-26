# Flux 1.dev VAE tokenizer used by the LucidFlux PiD decoder.
#
# Architecture: standard 2D image VAE with 16 latent channels, 8x spatial compression.
# Normalization uses scale_factor/shift_factor instead of external mean/std:
#   encode: z = scale_factor * (z_raw - shift_factor)
#   decode: z_raw = z / scale_factor + shift_factor
#
# The raw AutoEncoder code below is adapted from the official Flux repository
# (https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/autoencoder.py)
# with one change: Upsample.forward casts to float32 before interpolate (bfloat16 safety).

from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._ext.imaginaire.lazy_config import LazyDict
from pid._ext.imaginaire.utils import log
from pid._ext.imaginaire.utils.distributed import get_rank, sync_model_states
from pid._src.models.utils import load_state_dict
from pid._src.tokenizers.interface import VideoTokenizerInterface

__all__ = [
    "AutoEncoder",
    "FluxVAE",
    "FluxVAEInterface",
    "FluxVAEConfig",
]


# ===========================================================================
# Layer 1 — Raw Flux AutoEncoder (copied inline from official repo)
# ===========================================================================


@dataclass
class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 16
    scale_factor: float = 0.3611
    shift_factor: float = 0.1159


FLUX_VAE_PARAMS = AutoEncoderParams()


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h: torch.Tensor) -> torch.Tensor:
        B, C, H, W = h.shape
        q = self.q(h).reshape(B, 1, C, H * W).transpose(2, 3)
        k = self.k(h).reshape(B, 1, C, H * W).transpose(2, 3)
        v = self.v(h).reshape(B, 1, C, H * W).transpose(2, 3)
        h = F.scaled_dot_product_attention(q, k, v)
        return h.transpose(2, 3).reshape(B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj_out(self.attention(self.norm(x)))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = out_channels or in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(swish(self.norm1(x)))
        h = self.conv2(swish(self.norm2(h)))
        return self.nin_shortcut(x) + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 before interpolate for bfloat16 safety (matching Wan VAE convention)
        x = F.interpolate(x.float(), scale_factor=2.0, mode="nearest").type_as(x)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list,
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # downsampling
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = swish(self.norm_out(h))
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list,
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[-1]

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = swish(self.norm_out(h))
        h = self.conv_out(h)
        return h


class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        return mean


class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams = None):
        super().__init__()
        if params is None:
            params = FLUX_VAE_PARAMS
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        # sample=False for deterministic inference
        self.reg = DiagonalGaussian(sample=False)
        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def encode(self, x: torch.Tensor, useless: float = None) -> torch.Tensor:
        if x.ndim == 5:
            assert x.shape[2] == 1, f"Image-only VAE requires T=1, got T={x.shape[2]}"
            x = x.squeeze(2)  # (B, C, H, W)
            video_format_input = True
        else:
            video_format_input = False

        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)

        if video_format_input:
            z = z.unsqueeze(2)  # (B, 16, 1, h, w)

        return z

    def decode(self, z: torch.Tensor, useless: float = None) -> torch.Tensor:
        if z.ndim == 5:
            assert z.shape[2] == 1, f"Image-only VAE requires T=1, got T={z.shape[2]}"
            z = z.squeeze(2)  # (B, 16, h, w)
            video_format_input = True
        else:
            video_format_input = False

        z = z / self.scale_factor + self.shift_factor

        x = self.decoder(z)

        if video_format_input:
            x = x.unsqueeze(2)  # (B, 3, 1, H, W)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ===========================================================================
# Layer 2 — Factory function
# ===========================================================================


def _flux_vae(
    pretrained_path: str = None,
    device: str = "cpu",
    s3_credential_path: str = "credentials/s3_training.secret",
) -> AutoEncoder:
    """Build Flux AutoEncoder with optional checkpoint loading.

    Init pattern (safe for multi-rank sync):
      1. Build on meta device on all ranks.
      2. to_empty(device) on ALL ranks — materializes every param with the model's
         native dtype and standard contiguous strides.
      3. On rank 0, cast the ckpt tensors to the parameter dtype/device and copy
         them in with assign=False (strict=True). This preserves the allocated
         storage/dtype/layout from step 2 — no meta-tensor leakage, no dtype
         surprise from fp16/bf16 safetensors, no stride/contiguity quirks.
      4. sync_model_states broadcasts rank 0's values to everyone.

    The previous approach (load_state_dict(assign=True) + model.to(device)) would
    replace parameters with the ckpt tensors verbatim, so rank 0 could end up with
    a different dtype (e.g. fp16 safetensors) or different strides than the other
    ranks' to_empty() tensors — which silently deadlocks NCCL in sync_model_states.
    """
    params = FLUX_VAE_PARAMS

    with torch.device("meta"):
        model = AutoEncoder(params)

    model.to_empty(device=device)

    if pretrained_path is not None and get_rank() == 0:
        ckpt = load_state_dict(
            pretrained_path,
            s3_credential_path=s3_credential_path if pretrained_path.startswith("s3://") else None,
        )
        log.info(f"Loading Flux VAE from {pretrained_path}")
        model_state = model.state_dict()
        ckpt = {
            k: v.to(device=model_state[k].device, dtype=model_state[k].dtype)
            for k, v in ckpt.items()
            if k in model_state
        }
        model.load_state_dict(ckpt, strict=True)

    sync_model_states(model)

    return model


# ===========================================================================
# Layer 3 — FluxVAE wrapper (dtype / AMP handling)
# ===========================================================================


class FluxVAE:
    """Wrapper with dtype/AMP handling. All tensors are 4D (B, C, H, W).

    Unlike Wan VAE, Flux VAE does not need external mean/std — the scale_factor
    and shift_factor normalization is baked into AutoEncoder.encode/decode.
    """

    def __init__(
        self,
        vae_pth: str = "weights/flux-dev/ae.safetensors",
        s3_credential_path: str = "credentials/s3_training.secret",
        dtype: torch.dtype = torch.float,
        device: str = "cuda",
        is_amp: bool = True,
    ):
        self.dtype = dtype
        self.device = device
        self.scale = None

        self.model = _flux_vae(
            pretrained_path=vae_pth,
            device=device,
            s3_credential_path=s3_credential_path,
        )
        self.model = self.model.eval().requires_grad_(False)
        self.is_amp = is_amp
        if not is_amp:
            self.model = self.model.to(dtype=dtype)
            self.context = nullcontext()
        else:
            self.context = torch.amp.autocast("cuda", dtype=dtype)

    def count_param(self):
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) in [-1, 1]. Returns (B, 16, H/8, W/8)."""
        in_dtype = images.dtype
        with self.context:
            if not self.is_amp:
                images = images.to(self.dtype)
            latent = self.model.encode(images)
        return latent.to(in_dtype)

    @torch.no_grad()
    def decode(self, zs: torch.Tensor) -> torch.Tensor:
        """zs: (B, 16, h, w). Returns (B, 3, H, W)."""
        in_dtype = zs.dtype
        with self.context:
            if not self.is_amp:
                zs = zs.to(self.dtype)
            recon = self.model.decode(zs)
        return recon.to(in_dtype)


# ===========================================================================
# Layer 4 — FluxVAEInterface(VideoTokenizerInterface)
# ===========================================================================


class FluxVAEInterface(VideoTokenizerInterface):
    """Pipeline-compatible interface for Flux VAE. Image-only (temporal_compression_factor=1)."""

    def __init__(self, chunk_duration: int = 1, **kwargs):
        self.model = FluxVAE(
            dtype=torch.bfloat16,
            is_amp=False,
            vae_pth=kwargs.get("vae_pth", "weights/flux-dev/ae.safetensors"),
            s3_credential_path=kwargs.get("s3_credential_path", "credentials/s3_training.secret"),
        )
        self.chunk_duration = chunk_duration

    @property
    def dtype(self):
        return self.model.dtype

    def reset_dtype(self):
        pass

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Accept 5D (B,C,T,H,W) for pipeline compat. T must be 1. Returns 5D."""
        if state.ndim == 5:
            assert state.shape[2] == 1, f"Image-only VAE requires T=1, got T={state.shape[2]}"
            x = state.squeeze(2)  # (B, C, H, W)
        else:
            x = state

        latent = self.model.encode(x)  # (B, 16, h, w)
        return latent.unsqueeze(2)  # (B, 16, 1, h, w)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Accept 5D (B,C,T,H,W) for pipeline compat. T must be 1. Returns 5D."""
        if latent.ndim == 5:
            assert latent.shape[2] == 1, f"Image-only VAE requires T=1, got T={latent.shape[2]}"
            z = latent.squeeze(2)  # (B, 16, h, w)
        else:
            z = latent

        recon = self.model.decode(z)  # (B, 3, H, W)
        return recon.unsqueeze(2)  # (B, 3, 1, H, W)

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return num_pixel_frames  # No temporal compression

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        return num_latent_frames  # No temporal compression

    @property
    def spatial_compression_factor(self):
        return 8

    @property
    def temporal_compression_factor(self):
        return 1

    @property
    def pixel_chunk_duration(self):
        return self.chunk_duration

    @property
    def latent_chunk_duration(self):
        return self.chunk_duration

    @property
    def latent_ch(self):
        return 16

    @property
    def spatial_resolution(self):
        return 512

    @property
    def name(self):
        return "flux_vae_tokenizer"


# ===========================================================================
# Layer 5 — LazyDict config
# ===========================================================================

FluxVAEConfig: LazyDict = L(FluxVAEInterface)(
    name="flux_vae_tokenizer",
)
