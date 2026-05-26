# Model and network config registration for PixelDiT T2I (text-to-image pixel diffusion).
#
# Registers:
# - Model config: ddp_pixeldit
# - Network configs: pixeldit_stage3_1024px (and future variants)
#
# PixelDiT is a pixel-space MMDiT architecture for text-to-image generation.
# Unlike SSDD models which use a VAE latent as condition, PixelDiT operates
# directly on pixels with text embeddings from Gemma-2-2b-it as conditioning.
#
# Usage in experiment configs:
#   defaults=[
#       {"override /model": "ddp_pixeldit"},
#       {"override /net": "pixeldit_stage3_1024px"},
#   ]

from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._src.models.pixeldit_model import PixelDiTModel, PixelDiTModelConfig
from pid._src.networks.pixeldit_official import PixDiT_T2I

# =============================================================================
# Model config
# =============================================================================

DDP_PIXELDIT_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(PixelDiTModel)(
        config=PixelDiTModelConfig(
            precision="bfloat16",
        ),
        _recursive_=False,
    ),
)

# =============================================================================
# Network configs
# =============================================================================

PIXELDIT_FINETUNE_2048PX = L(PixDiT_T2I)(
    in_channels=3,
    num_groups=24,
    hidden_size=1536,
    pixel_hidden_size=16,
    pixel_attn_hidden_size=1152,
    pixel_num_groups=16,
    patch_depth=14,
    pixel_depth=2,
    patch_size=16,
    txt_embed_dim=2304,
    txt_max_length=300,
    use_text_rope=True,
    text_rope_theta=10000.0,
    repa_encoder_index=6,
    shift=6.0,
    rope_mode="ntk_aware",
    rope_ref_h=1024,
    rope_ref_w=1024,
)

# =============================================================================
# Registration functions
# =============================================================================


def register_model_pixeldit():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="ddp_pixeldit", node=DDP_PIXELDIT_CONFIG)


def register_pixeldit_net():
    cs = ConfigStore.instance()
    cs.store(group="net", package="model.config.net", name="pixeldit_finetune_2048px", node=PIXELDIT_FINETUNE_2048PX)
