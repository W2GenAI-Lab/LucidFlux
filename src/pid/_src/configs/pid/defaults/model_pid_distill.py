# Model config registration for PID distillation (inference subset).
#
# Registers `ddp_distill_pid` — the model node referenced by the
# SFT-distill experiments in pid/experiment/.
# Training-only optimizer / loss / discriminator fields are stripped.

from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._ext.imaginaire.lazy_config import LazyDict
from pid._src.models.pid_distill_model import PidDistillModel, PidDistillModelConfig

DDP_DISTILL_PID_CONFIG = LazyDict(
    dict(
        model=L(PidDistillModel)(
            config=PidDistillModelConfig(
                precision="bfloat16",
            ),
            _recursive_=False,
        ),
    ),
    flags={"allow_objects": True},
)


def register_model_pid_distill():
    cs = ConfigStore.instance()
    cs.store(
        group="model",
        package="_global_",
        name="ddp_distill_pid",
        node=DDP_DISTILL_PID_CONFIG,
    )
