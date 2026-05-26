# Conditioner config registration for PID (super-resolution).
#
# Handles caption string dropout + LQ tensor dropout (coupled for image and latent).
#
# Registers:
# - pid_caption_lq: caption (10% drop) + lq_video_or_image (10% drop) + lq_latent (10% drop, coupled)
# - pid_lq_only: caption (0% drop) + lq_video_or_image (10% drop) + lq_latent (10% drop, coupled)
#   When caption dropout=0, uncondition also keeps caption — CFG only applies to LQ.

from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._src.modules.conditioner import (
    CaptionStringDrop,
    LQTensorDrop,
    PidConditioner,
)

# Caption + LQ with 10% dropout each (full dual CFG)
Pid_CaptionLQ_Config = L(PidConditioner)(
    caption=L(CaptionStringDrop)(
        input_key="caption",
        output_key="caption",
        dropout_rate=0.1,
    ),
    lq_video_or_image=L(LQTensorDrop)(
        input_key="LQ_video_or_image",
        output_key="lq_video_or_image",
        dropout_rate=0.1,
        is_primary=True,
    ),
    lq_latent=L(LQTensorDrop)(
        input_key="LQ_latent",
        output_key="lq_latent",
        dropout_rate=0.1,
        is_primary=False,
    ),
)

# LQ-only CFG: caption never dropped, only LQ dropped for CFG
Pid_LQOnly_Config = L(PidConditioner)(
    caption=L(CaptionStringDrop)(
        input_key="caption",
        output_key="caption",
        dropout_rate=0.0,  # Never dropped -> uncondition also keeps caption
    ),
    lq_video_or_image=L(LQTensorDrop)(
        input_key="LQ_video_or_image",
        output_key="lq_video_or_image",
        dropout_rate=0.1,
        is_primary=True,
    ),
    lq_latent=L(LQTensorDrop)(
        input_key="LQ_latent",
        output_key="lq_latent",
        dropout_rate=0.1,
        is_primary=False,
    ),
)


def register_conditioner_pid():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="pid_caption_lq",
        node=Pid_CaptionLQ_Config,
    )
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="pid_lq_only",
        node=Pid_LQOnly_Config,
    )
