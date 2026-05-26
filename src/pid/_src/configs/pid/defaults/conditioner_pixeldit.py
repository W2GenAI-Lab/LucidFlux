# Conditioner config registration for PixelDiT T2I models.
#
# PixelDiT encodes text inside the model (Gemma-2-2b-it), so the conditioner
# only handles caption string dropout for CFG training — no pre-computed
# tensor embeddings needed.
#
# Registers:
# - pixeldit_caption: caption-only conditioner with 10% dropout
# - pixeldit_caption_drop20: caption-only conditioner with 20% dropout

from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._src.modules.conditioner import (
    CaptionStringDrop,
    PixelDiTConditioner,
)

# Caption-only conditioner with 10% dropout (matches original class_dropout_prob=0.1)
PixelDiTCaptionConfig = L(PixelDiTConditioner)(
    caption=L(CaptionStringDrop)(
        input_key="caption",
        output_key="caption",
        dropout_rate=0.1,
    ),
)


def register_conditioner_pixeldit():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="pixeldit_caption",
        node=PixelDiTCaptionConfig,
    )
