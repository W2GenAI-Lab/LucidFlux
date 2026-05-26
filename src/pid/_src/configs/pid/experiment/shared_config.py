# Shared "Chi-prompt" prefix used by the inference SFT-distill experiments.

_CHI_PROMPT = [
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
]

NEGATIVE_PROMPT = (
    "low quality, worst quality, over-saturated, three legs, six fingers, cartoon, anime, "
    "cgi, low res, blurry, deformed, distortion, duplicated limbs, plastic skin, jpeg artifacts, "
    "watermark"
)


def _common_model_overrides(*, state_ch: int):
    """Common model.config.* fields for the Flux PiD decoder experiments."""
    return dict(
        precision="bfloat16",
        input_data_key="image",
        input_caption_key="caption",
        use_fixed_prompt=False,
        text_encoder_name="gemma-2-2b-it",
        caption_channels=2304,
        y_norm=True,
        y_norm_scale_factor=0.01,
        model_max_length=300,
        chi_prompt=_CHI_PROMPT,
        fm_timescale=1000.0,
        logit_mean=0.0,
        logit_std=1.0,
        shift=6.0,
        cfg_scale=5.0,
        image_size=2048,
        negative_prompt=NEGATIVE_PROMPT,
        num_sample_steps=50,
        lq_condition_type="latent",
        state_ch=state_ch,
        student_sample_steps=4,
        student_t_list=[0.999, 0.866, 0.634, 0.342, 0.0],
        student_sample_type="sde",
        net=dict(
            train_lq_proj_only=False,
            lq_interval=2,
            rope_mode="ntk_aware",
            rope_ref_h=1024,
            rope_ref_w=1024,
            lq_gate_type="sigma_aware_per_token_per_dim",
        ),
    )
