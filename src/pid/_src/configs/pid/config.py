# Hydra config entry point for the PID inference subset.
#
# Registers the config groups required by the LucidFlux PiD decoder integration.

from typing import Any, List

import attrs

from pid._ext.imaginaire import config
from pid._ext.imaginaire.trainer import ImaginaireTrainer as Trainer
from pid._ext.imaginaire.utils.config_helper import import_all_modules_from_package
from pid._src.configs.pid.defaults.checkpoint import register_checkpoint
from pid._src.configs.pid.defaults.ckpt_type import register_ckpt_type
from pid._src.configs.pid.defaults.conditioner_pid import register_conditioner_pid
from pid._src.configs.pid.defaults.conditioner_pixeldit import register_conditioner_pixeldit
from pid._src.configs.pid.defaults.ema import register_ema
from pid._src.configs.pid.defaults.model_pid import (
    register_model_pid,
    register_pid_net,
)
from pid._src.configs.pid.defaults.model_pid_distill import (
    register_model_pid_distill,
)
from pid._src.configs.pid.defaults.model_pixeldit import (
    register_model_pixeldit,
    register_pixeldit_net,
)
from pid._src.configs.pid.defaults.tokenizer import register_tokenizer


@attrs.define(slots=False)
class Config(config.Config):
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"model": "ddp_distill_pid"},
            {"net": None},
            {"conditioner": None},
            {"ema": "power"},
            {"tokenizer": "flux_vae_tokenizer"},
            {"checkpoint": "local"},
            {"ckpt_type": "dummy"},
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    c.job.project = "pid"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.callbacks = None

    register_ema()
    register_tokenizer()
    register_checkpoint()
    register_ckpt_type()

    register_model_pixeldit()
    register_pixeldit_net()
    register_conditioner_pixeldit()

    register_model_pid()
    register_pid_net()
    register_conditioner_pid()
    register_model_pid_distill()

    import_all_modules_from_package("pid._src.configs.pid.experiment", reload=True)
    import_all_modules_from_package("pid._src.configs.pid.experiment_2kto4k", reload=True)
    return c
