# Shared overrides for the 2kto4k variant configs.
#
# The 2kto4k decoders were trained with multi-resolution data bucketing
# (2048→3840) plus a resolution-aware dynamic shift formula instead of the 2k
# variant's constant shift. At inference time `dynamic_shift` activates the
# resolution-aware shift computation in pixeldit_model (see the dynamic_shift
# precedence ladder in pid_distill_model.generate_samples_from_batch and the
# init-time log in PixelDiTModel.__init__).

from pid._src.configs.pid.experiment.shared_config import _common_model_overrides


def _common_model_overrides_2kto4k(*, state_ch: int) -> dict:
    """Drop-in for `_common_model_overrides` that adds `dynamic_shift`.

    `base_shift=4.0` + `base_image_size_for_shift_calc=1024` are the values
    the 2kto4k checkpoints were trained with — keep them in sync with the
    training configs under
    `linear-vsr/.../configs/pixel_diffusion_2k_to_4k/experiment_2k_to_4k/`.
    """
    cfg = _common_model_overrides(state_ch=state_ch)
    cfg["dynamic_shift"] = dict(
        base_shift=4.0,
        base_image_size_for_shift_calc=1024,
    )
    return cfg
