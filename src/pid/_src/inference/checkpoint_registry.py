# PiD checkpoint registry for the LucidFlux integration.
#
# LucidFlux only needs the Flux VAE PiD decoders. The rest of the upstream PiD
# demo registry is intentionally not vendored here.

from dataclasses import dataclass


@dataclass(frozen=True)
class PIDCheckpoint:
    experiment: str
    checkpoint_path: str
    pid_scale: int


_CKPT_ROOT = "weights/pid"

VALID_CKPT_TYPES = ("2k", "2kto4k")


PID_CHECKPOINT_REGISTRY: dict[tuple[str, str], PIDCheckpoint] = {
    ("flux", "2k"): PIDCheckpoint(
        experiment="PiD_res2k_sr4x_official_flux_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("flux", "2kto4k"): PIDCheckpoint(
        experiment="PiD_res2kto4k_sr4x_official_flux_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
}


def get_pid_checkpoint(backbone: str, ckpt_type: str = "2k") -> PIDCheckpoint:
    """Return the registered official PID checkpoint for `(backbone, ckpt_type)`.

    Raises KeyError with the list of valid keys when the pair is unknown.
    """
    if ckpt_type not in VALID_CKPT_TYPES:
        raise KeyError(f"Unknown ckpt_type {ckpt_type!r}. Valid: {VALID_CKPT_TYPES}")
    try:
        return PID_CHECKPOINT_REGISTRY[(backbone, ckpt_type)]
    except KeyError as exc:
        valid = ", ".join(sorted(f"{b}+{t}" for b, t in PID_CHECKPOINT_REGISTRY))
        raise KeyError(f"Unknown (backbone, ckpt_type)=({backbone!r}, {ckpt_type!r}). Valid: {valid}") from exc


__all__ = [
    "PIDCheckpoint",
    "PID_CHECKPOINT_REGISTRY",
    "VALID_CKPT_TYPES",
    "get_pid_checkpoint",
]
