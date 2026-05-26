import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image
from einops import rearrange
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from src.flux.sampling import denoise_lucidflux, get_noise, get_schedule, unpack
from src.flux.util import load_ae, load_flow_model
from src.flux.align_color import wavelet_reconstruction
from src.flux.lucidflux import (
    load_dual_condition_branch,
    load_lucidflux_weights,
    load_precomputed_embeddings,
    load_redux_image_encoder,
    load_siglip_model,
    load_swinir,
    move_modules_to_device,
    prepare_with_embeddings,
    preprocess_lq_image,
)
from src.flux.flux_prior_redux_ir import siglip_from_unit_tensor

def create_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to LucidFlux weights (.pth)"
    )
    parser.add_argument(
        "--control_image", type=str, required=True,
        help="Path to the input image or a directory of images for control"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)"
    )
    parser.add_argument(
        "--offload", action='store_true', help="Offload model to CPU when not in use"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/",
        help="The output directory where generation image is saved"
    )
    parser.add_argument(
        "--width", type=int, default=None, help="The width for generated image. If not specified, use original image size adjusted to multiple of 16"
    )
    parser.add_argument(
        "--height", type=int, default=None, help="The height for generated image. If not specified, use original image size adjusted to multiple of 16"
    )
    parser.add_argument(
        "--num_steps", type=int, default=50, help="The num_steps for diffusion process"
    )
    parser.add_argument(
        "--guidance", type=float, default=4, help="The guidance for diffusion process"
    )
    parser.add_argument(
        "--seed", type=int, default=123456789, help="A seed for reproducible inference"
    )
    parser.add_argument(
        "--swinir_pretrained", type=str, default=None, help="path to SwinIR checkpoint for prior"
    )
   
    parser.add_argument(
        "--siglip_ckpt", type=str, default="siglip2-so400m-patch16-512",
        help="HF id or path for SigLIP vision model"
    )
    return parser


def main(args):
    name = "flux-dev"
    offload = args.offload
    is_schnell = name == "flux-schnell"
    
    torch_device = torch.device(args.device)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # 使用预计算的embeddings
    embeddings_path = "weights/lucidflux/prompt_embeddings.pt"
    print(f"Loading precomputed embeddings from {embeddings_path}")
    embeddings_data = load_precomputed_embeddings(embeddings_path, torch_device)
    precomputed_txt = embeddings_data["txt"]
    precomputed_vec = embeddings_data["vec"]
    original_prompt = embeddings_data["prompt"]
    print(f"Loaded embeddings for prompt: '{original_prompt}'")
    print(f"txt shape: {precomputed_txt.shape}, vec shape: {precomputed_vec.shape}")

    # base models
    model = load_flow_model(name, device=torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)


    lucidflux_weights = load_lucidflux_weights(args.checkpoint)
    dual_condition_branch = load_dual_condition_branch(
        name=name,
        lucidflux_weights=lucidflux_weights,
        device=torch_device,
        offload=offload,
        branch_dtype=torch.bfloat16,
    )

    swinir = load_swinir(torch_device, args.swinir_pretrained, offload)

    dtype = torch.bfloat16 if torch_device.type == "cuda" else torch.float32
    siglip_model = load_siglip_model(args.siglip_ckpt, torch_device, dtype, offload)
    redux_image_encoder = load_redux_image_encoder("cpu" if offload else torch_device, dtype, lucidflux_weights["connector"])

    width = 16 * args.width // 16
    height = 16 * args.height // 16
    timesteps = get_schedule(
        args.num_steps,
        (width // 8) * (height // 8) // (16 * 16),
        shift=(not is_schnell),
    )

    # build image list
    if os.path.isdir(args.control_image):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
        input_paths = [
            os.path.join(args.control_image, f)
            for f in sorted(os.listdir(args.control_image))
            if os.path.isfile(os.path.join(args.control_image, f)) and f.lower().endswith(exts)
        ]
        if len(input_paths) == 0:
            raise ValueError(f"No image files found in directory: {args.control_image}")
    else:
        input_paths = [args.control_image]

    if len(input_paths) == 0:
        return

    from torchvision.utils import save_image

    # loop
    for img_path in input_paths:
        filename = os.path.basename(img_path).split(".")[0]
        
        # For each image, compute processed resolution and persist preview
        lq_processed = preprocess_lq_image(img_path, args.width, args.height)
        # lq_processed.save(os.path.join(args.output_dir, f"{filename}_lq_processed.jpeg"))
        condition_cond = torch.from_numpy((np.array(lq_processed) / 127.5) - 1)
        condition_cond = condition_cond.permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16).to(torch_device)
        condition_cond_pre = None

        with torch.no_grad():
            # SwinIR prior - 确保输入在正确的设备上
            ci_01 = torch.clamp((condition_cond.float() + 1.0) / 2.0, 0.0, 1.0)
            if offload:
                swinir.to(torch_device)
            ci_pre = swinir(ci_01.to(torch_device)).float().clamp(0.0, 1.0)
            if offload:
                swinir.to("cpu")
            # save_image(ci_pre, os.path.join(args.output_dir, f"{filename}_swinir_pre.jpeg"))
            condition_cond_pre = (ci_pre * 2.0 - 1.0).to(torch.bfloat16)

            # diffusion inputs
            torch.manual_seed(args.seed)
            x = get_noise(
                1, height, width, device=torch_device,
                dtype=torch.bfloat16, seed=args.seed
            )
            # 使用预计算的embeddings
            inp_cond = prepare_with_embeddings(
                img=x, precomputed_txt=precomputed_txt, precomputed_vec=precomputed_vec
            )

            # SigLIP feature -> Redux image embeds
            # Match preprocessing size to SigLIP config to avoid positional embedding mismatch
            siglip_size = getattr(getattr(siglip_model, "config", None), "image_size", 512)
            siglip_pixel_values_pre = siglip_from_unit_tensor(ci_pre, size=(siglip_size, siglip_size))
            inputs = {"pixel_values": siglip_pixel_values_pre.to(device=torch_device, dtype=dtype)}
            if offload:
                siglip_model.to(torch_device)
            siglip_image_pre_fts = siglip_model(**inputs).last_hidden_state.to(dtype=dtype)
            if offload:
                siglip_model.to("cpu")
                torch.cuda.empty_cache()
            enc_dtype = redux_image_encoder.redux_up.weight.dtype
            if offload:
                redux_image_encoder.to(torch_device)
            image_embeds = redux_image_encoder(
                siglip_image_pre_fts.to(device=torch_device, dtype=enc_dtype)
            )["image_embeds"]
            if offload:
                redux_image_encoder.to("cpu")
                torch.cuda.empty_cache()

            # concat to txt and extend txt_ids
            txt = inp_cond["txt"].to(device=torch_device, dtype=torch.bfloat16)
            txt_ids = inp_cond["txt_ids"].to(device=torch_device, dtype=torch.bfloat16)
            siglip_txt = torch.cat([txt, image_embeds.to(dtype=torch.bfloat16)], dim=1)
            B, L, C = txt_ids.shape
            extra_ids = torch.zeros((B, 1024, C), device=txt_ids.device, dtype=torch.bfloat16)
            siglip_txt_ids = torch.cat([txt_ids, extra_ids], dim=1).to(dtype=torch.bfloat16)

            # offload model (except main flow model)
            if offload:
                move_modules_to_device(torch_device, model, dual_condition_branch)
                torch.cuda.empty_cache()

            x = denoise_lucidflux(
                model,
                dual_condition_model=dual_condition_branch,
                img=inp_cond["img"],
                img_ids=inp_cond["img_ids"],
                txt=txt,
                txt_ids=txt_ids,
                siglip_txt=siglip_txt,
                siglip_txt_ids=siglip_txt_ids,
                vec=inp_cond["vec"],
                timesteps=timesteps,
                guidance=args.guidance,
                condition_cond_lq=condition_cond,
                condition_cond_pre=condition_cond_pre,
            )
            if offload:
                move_modules_to_device("cpu", model, dual_condition_branch)
                torch.cuda.empty_cache()
                ae.decoder.to(x.device)

            x = unpack(x.float(), height, width)
            x = ae.decode(x)
            if args.offload:
                ae.decoder.cpu()
                torch.cuda.empty_cache()

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())

        hq = wavelet_reconstruction((x1.permute(2, 0, 1) + 1.0) / 2, ci_pre.squeeze(0))
        hq = hq.clamp(0, 1)
        save_image(hq, os.path.join(args.output_dir, f"{filename}.png"))
        print(f"[INFO] {filename}  is done. Path: {args.output_dir}")
        

if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
