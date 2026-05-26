from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.lazy_config import LazyDict
from pid._src.configs.pid.experiment.shared_config import (
    _common_model_overrides,
)


def _flux_distill_experiment(name: str) -> LazyDict:
    return LazyDict(
        dict(
            defaults=[
                {"override /model": "ddp_distill_pid"},
                {"override /net": "pid_sr4x"},
                {"override /conditioner": "pid_caption_lq"},
                {"override /ckpt_type": "dcp"},
                {"override /ema": None},
                {"override /checkpoint": "local"},
                {"override /tokenizer": "flux_vae_tokenizer"},
                "_self_",
            ],
            job=dict(group="pid_official", name=name),
            model=dict(config=_common_model_overrides(state_ch=16)),
        ),
    )


# Single unified official inference experiment for the Flux1 VAE PID checkpoint.
# (zimage reuses the same model because ZImage's diffusers pipeline shares Flux1's
# 16-ch VAE — see checkpoint_registry.py.)
PID_RES2K_SR4X_OFFICIAL_FLUX_DISTILL_4STEP = _flux_distill_experiment("PiD_res2k_sr4x_official_flux_distill_4step")


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=PID_RES2K_SR4X_OFFICIAL_FLUX_DISTILL_4STEP["job"]["name"],
    node=PID_RES2K_SR4X_OFFICIAL_FLUX_DISTILL_4STEP,
)
