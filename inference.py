import os
import torch
import argparse
import numpy as np
from PIL import Image
from einops import rearrange

from src.flux.sampling import denoise_lucidflux, get_noise, get_schedule, prepare, unpack
from src.flux.util import (load_ae, load_clip, load_t5,
                           load_flow_model, load_single_condition_branch, load_safetensors)
from src.flux.swinir import SwinIR
import torch.nn as nn
from src.flux.align_color import wavelet_reconstruction

from transformers import SiglipVisionModel, AutoProcessor
from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from src.flux.flux_prior_redux_ir import siglip_from_unit_tensor
from typing import Optional
import math

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb

ACT2CLS = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}

def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    else:
        raise ValueError(f"activation function {act_fn} not found in ACT2FN mapping {list(ACT2CLS.keys())}")

class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample

class Modulation(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 2 * dim, bias=bias)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=dim)
        self.control_index_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=dim)

    def forward(self, x, timestep, control_index):
        timesteps_proj = self.time_proj(timestep * 1000)
        
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=x.dtype))  # (N, D)

        # Expand scalar control_index to batch dimension and project like timesteps (256-dim)
        if control_index.dim() == 0:
            control_index = control_index.repeat(x.shape[0])
        elif control_index.dim() == 1 and control_index.shape[0] != x.shape[0]:
            control_index = control_index.expand(x.shape[0])
        control_index = control_index.to(device=x.device, dtype=x.dtype)
        control_index_proj = self.time_proj(control_index)
        control_index_emb = self.control_index_embedder(control_index_proj.to(dtype=x.dtype))  # (N, D)
        timesteps_emb = timesteps_emb + control_index_emb
        emb = self.linear(self.silu(timesteps_emb))
        shift_msa, scale_msa = emb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x

class DualConditionBranch(nn.Module):
    def __init__(self, condition_branch_lq: nn.Module, condition_branch_ldr: nn.Module, modulation_lq: nn.Module, modulation_ldr: nn.Module):
        super().__init__()
        self.lq = condition_branch_lq
        self.ldr = condition_branch_ldr
        self.modulation_lq = modulation_lq
        self.modulation_ldr = modulation_ldr

    def forward(
        self,
        *,
        img,
        img_ids,
        condition_cond_lq,
        txt,
        txt_ids,
        y,
        timesteps,
        guidance,
        condition_cond_ldr=None,
    ):
        out_lq = self.lq(
            img=img,
            img_ids=img_ids,
            controlnet_cond=condition_cond_lq,
            txt=txt,
            txt_ids=txt_ids,
            y=y,
            timesteps=timesteps,
            guidance=guidance,
        )
        
        out_ldr = self.ldr(
            img=img,
            img_ids=img_ids,
            controlnet_cond=condition_cond_ldr,
            txt=txt,
            txt_ids=txt_ids,
            y=y,
            timesteps=timesteps,
            guidance=guidance,
        )
        out = []
        num_blocks = 19
        for i in range(num_blocks // 2 + 1):
            for control_index, (lq, ldr) in enumerate(zip(out_lq, out_ldr)):
                control_index = torch.tensor(control_index, device=timesteps.device, dtype=timesteps.dtype)
                lq = self.modulation_lq(lq, timesteps, i * 2 + control_index)

                if len(out) == num_blocks:
                    break

                ldr = self.modulation_ldr(ldr, timesteps, i * 2 + control_index)
                out.append(lq + ldr)
        return out


def create_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--control_image", type=str, required=True,
        help="Path to the input image or a directory of images for control"
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="The input text prompt"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)"
    )
    parser.add_argument(
        "--offload", action='store_true', help="Offload model to CPU when not in use"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/",
        help="The output directory where generation image is saved"
    )
    parser.add_argument(
        "--width", type=int, default=None, help="The width for generated image. If not specified, use original image size adjusted to multiple of 16"
    )
    parser.add_argument(
        "--height", type=int, default=None, help="The height for generated image. If not specified, use original image size adjusted to multiple of 16"
    )
    parser.add_argument(
        "--num_steps", type=int, default=50, help="The num_steps for diffusion process"
    )
    parser.add_argument(
        "--guidance", type=float, default=4, help="The guidance for diffusion process"
    )
    parser.add_argument(
        "--seed", type=int, default=123456789, help="A seed for reproducible inference"
    )
    parser.add_argument(
        "--swinir_pretrained", type=str, default=None, help="path to SwinIR checkpoint for prior"
    )
   
    parser.add_argument(
        "--siglip_ckpt", type=str, default="siglip2-so400m-patch16-512",
        help="HF id or path for SigLIP vision model"
    )
    return parser


def preprocess_lq_image(image_path: str, width: int = 512, height: int = 512):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((width, height))
    return image


def load_redux_image_encoder(device: torch.device, dtype: torch.dtype, redux_state_dict: str):
    redux_image_encoder = ReduxImageEncoder()
    redux_image_encoder.load_state_dict(redux_state_dict, strict=False)

    redux_image_encoder.eval()
    redux_image_encoder.to(device).to(dtype=dtype)
    return redux_image_encoder


def main(args):
    name = "flux-dev"
    offload = args.offload
    is_schnell = name == "flux-schnell"
    
    torch_device = torch.device(args.device)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # base models
    model, ae, t5, clip, condition_lq = (
        load_flow_model(name, device="cpu" if offload else torch_device),
        load_ae(name, device="cpu" if offload else torch_device),
        load_t5(torch_device, max_length=256 if is_schnell else 512),
        load_clip(torch_device),
        load_single_condition_branch(name, torch_device).to(torch.bfloat16),
    )
    model = model.to(torch_device)


    # load model checkpoint
    if '.safetensors' in args.checkpoint:
        checkpoint = load_safetensors(args.checkpoint)
    else:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')

    condition_lq.load_state_dict(checkpoint["condition_lq"], strict=False)
    condition_lq = condition_lq.to(torch_device)

    condition_ldr = load_single_condition_branch(name, torch_device).to(torch.bfloat16)
    condition_ldr.load_state_dict(checkpoint["condition_ldr"], strict=False)

    modulation_lq = Modulation(dim=3072).to(torch.bfloat16)
    modulation_lq.load_state_dict(checkpoint["modulation_lq"], strict=False)

    modulation_ldr = Modulation(dim=3072).to(torch.bfloat16)
    modulation_ldr.load_state_dict(checkpoint["modulation_ldr"], strict=False)

    dual_condition_branch = DualConditionBranch(
            condition_lq,
            condition_ldr,
            modulation_lq=modulation_lq,
            modulation_ldr=modulation_ldr,
        ).to(torch_device)

    # SwinIR prior (frozen)
    if args.swinir_pretrained is None:
        raise ValueError("SwinIR pretrained is not provided")
    swinir = SwinIR(
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2,
        sf=8,
        img_range=1.0,
        upsampler="nearest+conv",
        resi_connection="1conv",
        unshuffle=True,
        unshuffle_scale=8,
    )
    ckpt_obj = torch.load(args.swinir_pretrained, map_location="cpu")
    state = ckpt_obj.get("state_dict", ckpt_obj)
    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    swinir.load_state_dict(new_state, strict=False)
    swinir.eval()
    for p in swinir.parameters():
        p.requires_grad_(False)
    swinir = swinir.to(torch_device)

    # SigLIP + Redux encoders (frozen for inference)
    dtype = torch.bfloat16 if torch_device.type == 'cuda' else torch.float32
    siglip_model = SiglipVisionModel.from_pretrained(args.siglip_ckpt)
    siglip_model.eval()
    siglip_model.to(torch_device).to(dtype=dtype)
    redux_image_encoder = load_redux_image_encoder(torch_device, dtype, checkpoint["connector"])

    width = 16 * args.width // 16
    height = 16 * args.height // 16
    timesteps = get_schedule(
        args.num_steps,
        (width // 8) * (height // 8) // (16 * 16),
        shift=(not is_schnell),
    )

    # build image list
    if os.path.isdir(args.control_image):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
        input_paths = [
            os.path.join(args.control_image, f)
            for f in sorted(os.listdir(args.control_image))
            if os.path.isfile(os.path.join(args.control_image, f)) and f.lower().endswith(exts)
        ]
        if len(input_paths) == 0:
            raise ValueError(f"No image files found in directory: {args.control_image}")
    else:
        input_paths = [args.control_image]

    if len(input_paths) == 0:
        return

    from torchvision.utils import save_image

    # loop
    for img_path in input_paths:
        filename = os.path.basename(img_path).split(".")[0]
        
        # For each image, compute processed resolution and persist preview
        lq_processed = preprocess_lq_image(img_path, args.width, args.height)
        lq_processed.save(os.path.join(args.output_dir, f"{filename}_lq_processed.jpeg"))
        condition_cond = torch.from_numpy((np.array(lq_processed) / 127.5) - 1)
        condition_cond = condition_cond.permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16).to(torch_device)
        condition_cond_ldr = None

        with torch.no_grad():
            # SwinIR prior
            ci_01 = torch.clamp((condition_cond.float() + 1.0) / 2.0, 0.0, 1.0)
            ci_pre = swinir(ci_01).float().clamp(0.0, 1.0)
            # save_image(ci_pre, os.path.join(args.output_dir, f"{filename}_swinir_pre.jpeg"))
            condition_cond_ldr = (ci_pre * 2.0 - 1.0).to(torch.bfloat16)

            # diffusion inputs
            torch.manual_seed(args.seed)
            x = get_noise(
                1, height, width, device=torch_device,
                dtype=torch.bfloat16, seed=args.seed
            )
            if offload:
                t5, clip = t5.to(torch_device), clip.to(torch_device)
            inp_cond = prepare(t5=t5, clip=clip, img=x, prompt=args.prompt)

            # SigLIP feature -> Redux image embeds
            # Match preprocessing size to SigLIP config to avoid positional embedding mismatch
            siglip_size = getattr(getattr(siglip_model, "config", None), "image_size", 512)
            siglip_pixel_values_pre = siglip_from_unit_tensor(ci_pre, size=(siglip_size, siglip_size))
            inputs = {"pixel_values": siglip_pixel_values_pre.to(device=torch_device, dtype=dtype)}
            siglip_image_pre_fts = siglip_model(**inputs).last_hidden_state.to(dtype=dtype)
            enc_dtype = redux_image_encoder.redux_up.weight.dtype
            image_embeds = redux_image_encoder(
                siglip_image_pre_fts.to(device=torch_device, dtype=enc_dtype)
            )["image_embeds"]

            # concat to txt and extend txt_ids
            txt = inp_cond["txt"].to(device=torch_device, dtype=torch.bfloat16)
            txt_ids = inp_cond["txt_ids"].to(device=torch_device, dtype=torch.bfloat16)
            siglip_txt = torch.cat([txt, image_embeds.to(dtype=torch.bfloat16)], dim=1)
            B, L, C = txt_ids.shape
            extra_ids = torch.zeros((B, 1024, C), device=txt_ids.device, dtype=torch.bfloat16)
            siglip_txt_ids = torch.cat([txt_ids, extra_ids], dim=1).to(dtype=torch.bfloat16)

            # offload TEs
            if offload:
                t5, clip = t5.cpu(), clip.cpu()
                torch.cuda.empty_cache()
                model = model.to(torch_device)

            x = denoise_lucidflux(
                model,
                dual_condition_model=dual_condition_branch,
                img=inp_cond["img"],
                img_ids=inp_cond["img_ids"],
                txt=txt,
                txt_ids=txt_ids,
                siglip_txt=siglip_txt,
                siglip_txt_ids=siglip_txt_ids,
                vec=inp_cond["vec"],
                timesteps=timesteps,
                guidance=args.guidance,
                condition_cond_lq=condition_cond,
                condition_cond_ldr=condition_cond_ldr,
            )
            if offload:
                model.cpu()
                torch.cuda.empty_cache()
                ae.decoder.to(x.device)

            x = unpack(x.float(), height, width)
            x = ae.decode(x)
            if args.offload:
                ae.decoder.cpu()
                torch.cuda.empty_cache()

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())

        hq = wavelet_reconstruction((x1.permute(2, 0, 1) + 1.0) / 2, ci_pre.squeeze(0))
        hq = hq.clamp(0, 1)
        save_image(hq, os.path.join(args.output_dir, f"{filename}_result.jpeg"))
        print(f"[INFO] {filename}  is done. Path: {args.output_dir}")
        

if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
