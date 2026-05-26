import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pid._src.inference.checkpoint_registry import get_pid_checkpoint
from pid._src.utils.model_loader import load_model_from_checkpoint


DEFAULT_PID_CONFIG_FILE = "src/pid/_src/configs/pid/config.py"
DEFAULT_PID_CAPTION_EMBEDDINGS = "weights/pid/gemma_prompt_embedding.pt"


def load_caption_embeddings(path: str | None) -> dict | None:
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"PiD prompt embedding not found: {path}. "
            "Download weights again or pass --pid_caption_embeddings ''."
        )
    data = torch.load(path, map_location="cpu")
    if "caption_embs" not in data:
        raise KeyError(f"{path} does not contain 'caption_embs'")
    return data


def save_tensor_image(sample: torch.Tensor, save_path: str, quality: int = 100, save_jpg_copy: bool = True) -> str:
    if sample.dim() == 4:
        sample = sample.squeeze(0) if sample.shape[0] == 1 else sample.squeeze(1)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    arr = ((sample.float().clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    img = Image.fromarray(arr)
    if save_path.lower().endswith((".jpg", ".jpeg")):
        img.save(save_path, quality=quality, subsampling=0)
    else:
        img.save(save_path)
        if save_jpg_copy:
            img.save(str(Path(save_path).with_suffix(".jpg")), quality=quality, subsampling=0)
    return save_path


class PiDFluxDecoder:
    def __init__(
        self,
        *,
        ckpt_type: str = "2kto4k",
        config_file: str = DEFAULT_PID_CONFIG_FILE,
        experiment: str | None = None,
        checkpoint_path: str | None = None,
        load_ema_to_reg: bool = False,
        scale: int | None = None,
        caption_embeddings_path: str | None = DEFAULT_PID_CAPTION_EMBEDDINGS,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("PiD decode currently requires CUDA.")

        default_ckpt = get_pid_checkpoint("flux", ckpt_type)
        self.experiment = experiment or default_ckpt.experiment
        self.checkpoint_path = checkpoint_path or default_ckpt.checkpoint_path
        self.scale = scale or default_ckpt.pid_scale
        self.caption_embeddings = load_caption_embeddings(caption_embeddings_path)
        experiment_opts = []
        if self.caption_embeddings is not None:
            experiment_opts.append(f"model.config.precomputed_caption_embeddings_path={caption_embeddings_path}")

        print(f"Loading PiD decoder from {self.checkpoint_path}")
        self.model, _ = load_model_from_checkpoint(
            experiment_name=self.experiment,
            checkpoint_path=self.checkpoint_path,
            config_file=config_file,
            enable_fsdp=False,
            experiment_opts=experiment_opts,
            strict=False,
            load_ema_to_reg=load_ema_to_reg,
        )
        self.model.eval()

    @torch.no_grad()
    def decode(
        self,
        latent: torch.Tensor,
        *,
        caption: str,
        cfg_scale: float = 1.0,
        num_steps: int | None = 4,
        seed: int = 123456789,
        shift: float | None = None,
        degrade_sigma: float = 0.0,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        latent = latent.to(device="cuda", dtype=torch.bfloat16)
        latent_h, latent_w = latent.shape[-2:]
        vae_h, vae_w = latent_h * 8, latent_w * 8
        target_hw = (vae_h * self.scale, vae_w * self.scale)

        if self.caption_embeddings is not None and self.caption_embeddings.get("caption") != caption:
            raise ValueError(
                "Loaded precomputed PiD caption embeddings for "
                f"{self.caption_embeddings.get('caption')!r}, but decode caption is {caption!r}. "
                "Regenerate the embeddings or pass --pid_caption_embeddings ''."
            )

        data_batch = {
            self.model.config.input_caption_key: [caption],
            "LQ_video_or_image": torch.zeros(1, 3, vae_h, vae_w, device="cuda", dtype=torch.bfloat16),
            "LQ_latent": latent,
            "degrade_sigma": torch.tensor([float(degrade_sigma)], device="cuda", dtype=torch.float32),
        }
        if self.caption_embeddings is not None and self.caption_embeddings.get("caption") == caption:
            data_batch["caption_embs"] = self.caption_embeddings["caption_embs"]
        samples = self.model.generate_samples_from_batch(
            data_batch,
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            seed=seed,
            shift=shift,
            image_size=target_hw,
        )
        return samples[0].float().cpu().clamp(-1, 1), target_hw


def align_color(
    pid_output: torch.Tensor,
    ci_pre: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    from src.flux.align_color import wavelet_decomposition

    strength = max(0.0, min(1.0, float(strength)))
    pid_01 = pid_output.squeeze(1) if pid_output.dim() == 4 and pid_output.shape[1] == 1 else pid_output
    pid_01 = (pid_01 + 1.0) / 2.0
    if pid_01.dim() == 3:
        pid_01 = pid_01.unsqueeze(0)

    ci_pre = ci_pre.to(device=pid_01.device, dtype=pid_01.dtype)
    ci_pre = F.interpolate(ci_pre, size=pid_01.shape[-2:], mode="bilinear", align_corners=False)
    if ci_pre.dim() == 3:
        ci_pre = ci_pre.unsqueeze(0)

    pid_high, pid_low = wavelet_decomposition(pid_01)
    _, ci_pre_low = wavelet_decomposition(ci_pre)
    aligned = (pid_high + ci_pre_low * strength + pid_low * (1.0 - strength)).clamp(0, 1)
    return aligned * 2.0 - 1.0
