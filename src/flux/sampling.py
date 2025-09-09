import math
from typing import Callable, Union

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None, 
    neg_image_proj: Tensor = None, 
    ip_scale: Union[Tensor, float] = 1.0,
    neg_ip_scale: Union[Tensor, float] = 1.0
):
    i = 0
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            image_proj=image_proj,
            ip_scale=ip_scale, 
        )
        if i >= timestep_to_start_cfg:
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec, 
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale, 
            )     
            pred = neg_pred + true_gs * (pred - neg_pred)
        img = img + (t_prev - t_curr) * pred
        i += 1
    return img


def denoise_controlnet(
    model: Flux,
    controlnet:None,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    siglip_txt: Tensor,
    siglip_txt_ids: Tensor,
    vec: Tensor,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    controlnet_gs=0.7,
    # dual-control optional params
    controlnet_cond_pre=None,
    controlnet_w_lq: float = 1.0,
    controlnet_w_pre: float = 1.0,
):
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        # align dtypes to model/controlnet precision (e.g., bf16)
        dtype = img.dtype
        img_ids_in = img_ids.to(dtype)
        txt_in = txt.to(dtype)
        txt_ids_in = txt_ids.to(dtype)
        vec_in = vec.to(dtype)
        cond_lq = controlnet_cond.to(dtype)
        cond_pre = controlnet_cond_pre.to(dtype) if controlnet_cond_pre is not None else None

        # support single or dual control
        if controlnet_cond_pre is not None:
            try:
                # try DualControlNet-style forward with pre-branch
                block_res_samples = controlnet(
                    img=img,
                    img_ids=img_ids_in,
                    controlnet_cond=cond_lq,
                    txt=txt_in,
                    txt_ids=txt_ids_in,
                    y=vec_in,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    controlnet_cond_pre=cond_pre,
                    w_lq=controlnet_w_lq,
                    w_pre=controlnet_w_pre,
                )
            except TypeError:
                # fallback: call twice and fuse
                out_lq = controlnet(
                    img=img,
                    img_ids=img_ids_in,
                    controlnet_cond=cond_lq,
                    txt=txt_in,
                    txt_ids=txt_ids_in,
                    y=vec_in,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
                out_pre = controlnet(
                    img=img,
                    img_ids=img_ids_in,
                    controlnet_cond=cond_pre,
                    txt=txt_in,
                    txt_ids=txt_ids_in,
                    y=vec_in,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
                block_res_samples = [controlnet_w_lq * a + controlnet_w_pre * b for a, b in zip(out_lq, out_pre)]
        else:
            block_res_samples = controlnet(
                        img=img,
                        img_ids=img_ids_in,
                        controlnet_cond=cond_lq,
                        txt=txt_in,
                        txt_ids=txt_ids_in,
                        y=vec_in,
                        timesteps=t_vec,
                        guidance=guidance_vec,
                    )
        if siglip_txt is not None:
            pred = model(
                img=img,
                img_ids=img_ids_in,
                txt=siglip_txt,
                txt_ids=siglip_txt_ids,
                y=vec_in,
                timesteps=t_vec,
                guidance=guidance_vec,
                block_controlnet_hidden_states=[i * controlnet_gs for i in block_res_samples]
            )
        else:      
            pred = model(
                img=img,
                img_ids=img_ids_in,
                txt=txt_in,
                txt_ids=txt_ids_in,
                y=vec_in,
                timesteps=t_vec,
                guidance=guidance_vec,
                block_controlnet_hidden_states=[i * controlnet_gs for i in block_res_samples]
            )
        

        img = img + (t_prev - t_curr) * pred
        #if use_gs:
        #    img = torch.cat([img] * 2)

    return img
    
def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
