import argparse
import logging
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from einops import rearrange
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from image_datasets.lq_gt_dataset import loader
from src.flux.flux_prior_redux_ir import siglip_from_unit_tensor
from src.flux.lucidflux import (
    export_lucidflux_state,
    load_condition_branch_with_redux,
    load_lucidflux_weights,
    load_precomputed_embeddings,
    load_siglip_model,
    load_swinir,
    prepare_with_embeddings,
)
from src.flux.util import load_ae, load_flow_model2


if hasattr(torch, "load"):
    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load


logger = get_logger(__name__, log_level="INFO")



class FlowMatchEulerDiscreteScheduler:
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 3.0):
        self.config = type(
            "Config",
            (),
            {"num_train_timesteps": num_train_timesteps, "shift": shift},
        )()
        timesteps = np.linspace(1, 0, num_train_timesteps)
        self.timesteps = torch.from_numpy(timesteps).float()
        self.sigmas = torch.from_numpy(timesteps).float()


def get_noisy_model_input_and_timesteps(
    noise_scheduler,
    latents: torch.Tensor,
    noise: torch.Tensor,
    device: torch.device,
    timestep_sampling: str = "shift",
    discrete_flow_shift: float = 3.1582,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = latents.shape[0]
    num_timesteps = noise_scheduler.config.num_train_timesteps

    if timestep_sampling == "shift":
        sigmas = torch.sigmoid(torch.randn((batch_size,), device=device) + discrete_flow_shift)
    else:
        raise ValueError(f"Unsupported timestep_sampling {timestep_sampling!r}. Expected 'shift'.")
    timesteps = sigmas * num_timesteps
    sigmas = sigmas.view(-1, 1, 1)
    noisy_model_input = (1 - sigmas) * latents + sigmas * noise
    return noisy_model_input, timesteps.long(), sigmas.squeeze()


def unpack_latents(x: torch.Tensor, packed_latent_height: int, packed_latent_width: int) -> torch.Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=packed_latent_height,
        w=packed_latent_width,
        ph=2,
        pw=2,
    )


def compute_loss(model_pred: torch.Tensor, target: torch.Tensor, loss_type: str = "l2") -> torch.Tensor:
    if loss_type == "l2":
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        return loss.mean(dim=(1, 2, 3)).mean()

    raise ValueError(f"Unsupported loss_type {loss_type!r}. Expected 'l2'.")


def parse_args():
    parser = argparse.ArgumentParser(description="LucidFlux training script")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parsed_args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(parsed_args.config)
    overrides = [arg[2:] if arg.startswith("--") else arg for arg in unknown_args]
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_cli(overrides))
    return config


def configure_logging(accelerator: Accelerator) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
        return

    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()


def get_models(name: str, device: torch.device, offload: bool):
    model = load_flow_model2(name, device="cpu")
    vae = load_ae(name, device="cpu" if offload else device)
    return model, vae


def create_optimizer(args, parameters):
    optimizer_type = args.get("optimizer_type", "adafactor").lower()
    if optimizer_type != "adafactor":
        raise ValueError(f"Unsupported optimizer_type {optimizer_type!r}. Expected 'adafactor'.")

    from transformers import Adafactor

    return Adafactor(
        parameters,
        lr=args.learning_rate,
        relative_step=args.get("relative_step", False),
        scale_parameter=args.get("scale_parameter", False),
        warmup_init=args.get("warmup_init", False),
    )


def cleanup_old_checkpoints(output_dir: str, checkpoints_total_limit: int | None):
    if checkpoints_total_limit is None:
        return

    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    checkpoints = sorted(checkpoints, key=lambda name: int(name.split("-")[1]))
    if len(checkpoints) < checkpoints_total_limit:
        return

    num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
    for checkpoint_name in checkpoints[:num_to_remove]:
        checkpoint_path = os.path.join(output_dir, checkpoint_name)
        logger.info("Removing old checkpoint %s", checkpoint_path)
        shutil.rmtree(checkpoint_path)


def resolve_resume_checkpoint(output_dir: str, resume_from_checkpoint: str | None):
    if not resume_from_checkpoint:
        return None
    if resume_from_checkpoint != "latest":
        return os.path.basename(resume_from_checkpoint)

    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    checkpoints = sorted(checkpoints, key=lambda name: int(name.split("-")[1]))
    return checkpoints[-1] if checkpoints else None


def validate_effective_training_config(args) -> None:
    optimizer_type = args.get("optimizer_type", "adafactor").lower()
    if optimizer_type != "adafactor":
        raise ValueError(f"optimizer_type must be 'adafactor', got {optimizer_type!r}")

    timestep_sampling = args.get("timestep_sampling", "shift")
    if timestep_sampling != "shift":
        raise ValueError(f"timestep_sampling must be 'shift', got {timestep_sampling!r}")

    loss_type = args.get("loss_type", "l2")
    if loss_type != "l2":
        raise ValueError(f"loss_type must be 'l2', got {loss_type!r}")


def get_weight_dtype(accelerator: Accelerator) -> torch.dtype:
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def prepare_control_image_features(
    control_image: torch.Tensor,
    swinir,
    siglip_model,
    siglip_dtype: torch.dtype,
    device: torch.device,
):
    control_image_pre = control_image
    siglip_image_pre_fts = None

    if swinir is None:
        return control_image_pre, siglip_image_pre_fts

    control_image_01 = torch.clamp((control_image.float() + 1.0) / 2.0, 0.0, 1.0)
    control_image_pre_01 = swinir(control_image_01).detach().clamp(0.0, 1.0)
    control_image_pre = control_image_pre_01 * 2.0 - 1.0

    siglip_size = getattr(getattr(siglip_model, "config", None), "image_size", 512)
    siglip_pixel_values = siglip_from_unit_tensor(
        control_image_pre_01,
        size=(siglip_size, siglip_size),
    )
    siglip_image_pre_fts = siglip_model(
        pixel_values=siglip_pixel_values.to(device=device, dtype=siglip_dtype)
    ).last_hidden_state
    return control_image_pre, siglip_image_pre_fts


def encode_batch_inputs(
    img: torch.Tensor,
    control_image: torch.Tensor,
    vae,
    precomputed_txt: torch.Tensor,
    precomputed_vec: torch.Tensor,
    swinir,
    siglip_model,
    siglip_dtype: torch.dtype,
    device: torch.device,
):
    latents = vae.encode(img.to(torch.float32))
    _, _, latent_height, latent_width = latents.shape
    packed_latent_height = latent_height // 2
    packed_latent_width = latent_width // 2

    model_inputs = prepare_with_embeddings(
        img=latents,
        precomputed_txt=precomputed_txt,
        precomputed_vec=precomputed_vec,
    )
    packed_latents = rearrange(
        latents,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=2,
        pw=2,
    )
    control_image_pre, siglip_image_pre_fts = prepare_control_image_features(
        control_image,
        swinir,
        siglip_model,
        siglip_dtype,
        device,
    )
    return (
        packed_latents,
        model_inputs,
        packed_latent_height,
        packed_latent_width,
        control_image_pre,
        siglip_image_pre_fts,
    )


def resume_if_needed(accelerator: Accelerator, args, num_update_steps_per_epoch: int):
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    resume_path = resolve_resume_checkpoint(args.output_dir, args.get("resume_from_checkpoint"))
    if not args.get("resume_from_checkpoint"):
        return global_step, first_epoch, initial_global_step

    if resume_path is None:
        accelerator.print(
            f"Checkpoint {args.resume_from_checkpoint!r} does not exist. Starting a new training run."
        )
        return global_step, first_epoch, initial_global_step

    accelerator.print(f"Resuming from checkpoint {resume_path}")
    accelerator.load_state(os.path.join(args.output_dir, resume_path))
    global_step = int(resume_path.split("-")[1])
    initial_global_step = global_step
    first_epoch = global_step // num_update_steps_per_epoch
    return global_step, first_epoch, initial_global_step


def main():
    args = parse_args()
    validate_effective_training_config(args)

    data_config = args.data_config
    train_batch_size = data_config.train_batch_size
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.get("mixed_precision", "bf16"),
        log_with=args.get("report_to", "wandb"),
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )
    configure_logging(accelerator)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.get("seed") is not None:
        set_seed(args.seed)

    device = accelerator.device
    siglip_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    dit, vae = get_models(name=args.model_name, device=device, offload=False)
    dit.requires_grad_(False)
    dit.to(device)
    if getattr(dit, "_set_gradient_checkpointing", None) is not None:
        dit._set_gradient_checkpointing(dit, True)

    vae.requires_grad_(False)
    vae.eval()

    prompt_embeddings = load_precomputed_embeddings(
        args.get("prompt_embeddings_path", "weights/lucidflux/prompt_embeddings.pt"),
        device,
    )
    precomputed_txt = prompt_embeddings["txt"]
    precomputed_vec = prompt_embeddings["vec"]

    lucidflux_weights = load_lucidflux_weights(
        args.get("lucidflux_weights_path", "weights/lucidflux/lucidflux.pth")
    )
    condition_branch = load_condition_branch_with_redux(
        name=args.model_name,
        lucidflux_weights=lucidflux_weights,
        device=device,
        offload=False,
        branch_dtype=torch.float32,
        connector_dtype=torch.float32,
    )
    condition_branch.train()

    siglip_model = load_siglip_model(
        args.get("siglip_ckpt", "weights/siglip"),
        device,
        siglip_dtype,
        offload=False,
    )
    siglip_model.requires_grad_(False)
    siglip_model.eval()

    swinir_ckpt = args.get("swinir_pretrained")
    swinir = load_swinir(device, swinir_ckpt, offload=False) if swinir_ckpt else None

    trainable_parameters = [param for param in condition_branch.parameters() if param.requires_grad]
    optimizer = create_optimizer(args, trainable_parameters)
    logger.info(
        "Trainable params: %.2f M",
        sum(param.numel() for param in trainable_parameters) / 1_000_000,
    )

    train_dataloader = loader(**OmegaConf.to_container(data_config, resolve=True))
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.get("max_train_steps") is None:
        raise ValueError("max_train_steps must be set in the training config")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    condition_branch, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        condition_branch, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = get_weight_dtype(accelerator)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            args.tracker_project_name,
            {"config": OmegaConf.to_container(args, resolve=True)},
        )

    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=args.get("discrete_flow_shift", 3.1582),
    )
    total_batch_size = train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running LucidFlux training *****")
    logger.info("  Num Epochs = %s", num_train_epochs)
    logger.info("  Instantaneous batch size per device = %s", train_batch_size)
    logger.info(
        "  Total train batch size (parallel, distributed & accumulation) = %s",
        total_batch_size,
    )
    logger.info("  Gradient Accumulation steps = %s", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %s", args.max_train_steps)
    logger.info("  Learning rate = %s", args.learning_rate)
    logger.info("  Optimizer = %s", args.get("optimizer_type", "adafactor"))
    logger.info("  Timestep sampling = %s", args.get("timestep_sampling", "shift"))
    logger.info("  Loss type = %s", args.get("loss_type", "l2"))

    global_step, first_epoch, initial_global_step = resume_if_needed(
        accelerator,
        args,
        num_update_steps_per_epoch,
    )
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for batch in train_dataloader:
            with accelerator.accumulate(condition_branch):
                img, control_image = batch
                img = img.to(device)
                control_image = control_image.to(device)

                with torch.no_grad():
                    (
                        packed_latents,
                        model_inputs,
                        packed_latent_height,
                        packed_latent_width,
                        control_image_pre,
                        siglip_image_pre_fts,
                    ) = encode_batch_inputs(
                        img,
                        control_image,
                        vae,
                        precomputed_txt,
                        precomputed_vec,
                        swinir,
                        siglip_model,
                        siglip_dtype,
                        device,
                    )

                batch_size = img.shape[0]
                noise = torch.randn_like(packed_latents)
                noisy_latents, timesteps, _ = get_noisy_model_input_and_timesteps(
                    noise_scheduler,
                    packed_latents,
                    noise,
                    device,
                    timestep_sampling=args.get("timestep_sampling", "shift"),
                    discrete_flow_shift=args.get("discrete_flow_shift", 3.1582),
                )
                guidance = torch.full(
                    (batch_size,),
                    args.get("guidance_scale", 1.0),
                    device=device,
                    dtype=weight_dtype,
                )
                normalized_timesteps = (timesteps.float() / 1000).to(weight_dtype)

                block_res_samples, siglip_txt, siglip_txt_ids = condition_branch(
                    img=noisy_latents.to(weight_dtype),
                    img_ids=model_inputs["img_ids"].to(weight_dtype),
                    condition_cond_lq=control_image.to(weight_dtype),
                    txt=model_inputs["txt"].to(weight_dtype),
                    txt_ids=model_inputs["txt_ids"].to(weight_dtype),
                    y=model_inputs["vec"].to(weight_dtype),
                    timesteps=normalized_timesteps,
                    guidance=guidance,
                    condition_cond_pre=control_image_pre.to(weight_dtype) if swinir is not None else None,
                    siglip_image_pre_fts=siglip_image_pre_fts if swinir is not None else None,
                )
                model_pred = dit(
                    img=noisy_latents.to(weight_dtype),
                    img_ids=model_inputs["img_ids"].to(weight_dtype),
                    txt=siglip_txt.to(weight_dtype),
                    txt_ids=siglip_txt_ids.to(weight_dtype),
                    block_controlnet_hidden_states=[
                        sample.to(dtype=weight_dtype) for sample in block_res_samples
                    ],
                    y=model_inputs["vec"].to(weight_dtype),
                    timesteps=normalized_timesteps,
                    guidance=guidance,
                )

                model_pred = unpack_latents(model_pred, packed_latent_height, packed_latent_width)
                target = unpack_latents(noise - packed_latents, packed_latent_height, packed_latent_width)
                loss = compute_loss(model_pred, target, loss_type=args.get("loss_type", "l2"))

                avg_loss = accelerator.gather(loss.repeat(batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.get("max_grad_norm", 0.0) > 0:
                    accelerator.clip_grad_norm_(condition_branch.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step_loss": loss.detach().item(),
                    },
                    step=global_step,
                )
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0 or global_step == 10:
                    if accelerator.is_main_process:
                        cleanup_old_checkpoints(args.output_dir, args.get("checkpoints_total_limit"))

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(condition_branch)
                        torch.save(
                            export_lucidflux_state(unwrapped_model),
                            os.path.join(save_path, "lucidflux.pth"),
                        )
                        logger.info("Saved state to %s (lucidflux.pth)", save_path)

            progress_bar.set_postfix(
                step_loss=loss.detach().item(),
                lr=lr_scheduler.get_last_lr()[0],
            )
            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
