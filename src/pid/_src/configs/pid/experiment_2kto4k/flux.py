from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.lazy_config import LazyDict
from pid._src.configs.pid.experiment_2kto4k.shared_config import (
    _common_model_overrides_2kto4k,
)


def _flux_2kto4k_distill_experiment(name: str) -> LazyDict:
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
            model=dict(config=_common_model_overrides_2kto4k(state_ch=16)),
        ),
    )


# Single unified official inference experiment for the Flux1 VAE PiD 2kto4k checkpoint.
# (zimage reuses the same model because ZImage's diffusers pipeline shares Flux1's
# 16-ch VAE — see checkpoint_registry.py.)
PID_RES2KTO4K_SR4X_OFFICIAL_FLUX_DISTILL_4STEP = _flux_2kto4k_distill_experiment(
    "PiD_res2kto4k_sr4x_official_flux_distill_4step"
)


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=PID_RES2KTO4K_SR4X_OFFICIAL_FLUX_DISTILL_4STEP["job"]["name"],
    node=PID_RES2KTO4K_SR4X_OFFICIAL_FLUX_DISTILL_4STEP,
)
