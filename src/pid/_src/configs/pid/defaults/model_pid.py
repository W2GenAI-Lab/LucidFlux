# Model and network config registration for PID (PixelDiT super-resolution).
#
# Registers:
# - Model config: ddp_pid
# - Network config: pid_sr4x (base controlnet, override params in experiments)
#
# PID extends the T2I PixelDiT architecture with LQ image/latent conditioning.
# The base T2I weights can be loaded with strict=False.
#
# Usage in experiment configs:
#   defaults=[
#       {"override /model": "ddp_pid"},
#       {"override /net": "pid_sr4x"},
#       {"override /conditioner": "pid_caption_lq"},
#   ]
#   # Then override net params per experiment, e.g.:
#   model=dict(config=dict(net=dict(lq_latent_channels=16)))

from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._src.models.pid_model import PidModel, PidModelConfig
from pid._src.networks.pid_net import PidNet

# =============================================================================
# Model config
# =============================================================================

DDP_PID_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(PidModel)(
        config=PidModelConfig(
            precision="bfloat16",
        ),
        _recursive_=False,
    ),
)

# =============================================================================
# Network config — single base, override in experiments
# =============================================================================

# Base PidNet network (controlnet injection — the only mode supported here).
# Experiments override: lq_in_channels, lq_latent_channels, lq_gate_type,
#                       lq_interval, train_lq_proj_only, etc.
PID_SR4X = L(PidNet)(
    # T2I backbone args (same as pixeldit_stage3_1024px)
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
    # SR-specific defaults (controlnet + latent-only)
    lq_inject_mode="controlnet",
    lq_in_channels=0,
    lq_latent_channels=16,
    lq_hidden_dim=512,
    lq_gate_type="sigma_aware_per_token_per_dim",
    lq_interval=1,
    zero_init_lq=True,
    train_lq_proj_only=False,
    sr_scale=4,
    # PiT LQ injection (disabled by default for backward compat)
    pit_lq_inject=False,
    pit_lq_gate_type="sigma_aware_per_token_per_dim",
)


# =============================================================================
# Registration
# =============================================================================


def register_model_pid():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="ddp_pid", node=DDP_PID_CONFIG)


def register_pid_net():
    cs = ConfigStore.instance()
    cs.store(
        group="net",
        package="model.config.net",
        name="pid_sr4x",
        node=PID_SR4X,
    )
