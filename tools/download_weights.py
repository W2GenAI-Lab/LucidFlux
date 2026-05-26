import argparse
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


FLUX_REPO = "black-forest-labs/FLUX.1-dev"
FLOW_FILE = "flux1-dev.safetensors"
AE_FILE = "ae.safetensors"
SWINIR_REPO = "lxq007/DiffBIR"
SWINIR_FILE = "general_swinir_v1.ckpt"
LUCIDFLUX_REPO = "W2GenAI/LucidFlux"
LUCIDFLUX_FILE = "lucidflux.pth"
PROMPT_EMBEDDINGS_FILE = "prompt_embeddings.pt"
PID_PROMPT_EMBEDDING_FILE = "gemma_prompt_embedding.pt"
ULTRAFLUX_REPO = "Owen777/UltraFlux-v1"
SIGLIP_REPO = "google/siglip2-so400m-patch16-512"
PID_REPO = "nvidia/PiD"
PID_FLUX_2K_EXPERIMENT = "PiD_res2k_sr4x_official_flux_distill_4step"
PID_FLUX_2KTO4K_EXPERIMENT = "PiD_res2kto4k_sr4x_official_flux_distill_4step"
PID_CHECKPOINT_FILE = "model_ema_bf16.pth"
MODEL_KEY = "flux-dev"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download fixed LucidFlux weights")
    parser.add_argument("--dest", type=str, default="weights", help="Destination root directory")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    parser.add_argument("--print-env", action="store_true", help="Print FLOW/AE export lines")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    return parser.parse_args()


def plan(dest_root: Path) -> dict[str, Path]:
    model_dir = dest_root / MODEL_KEY
    return {
        "flow": model_dir / FLOW_FILE,
        "ae": model_dir / AE_FILE,
        "env": dest_root / "env.sh",
        "manifest": dest_root / "manifest.json",
        "swinir": dest_root / "swinir.pth",
        "lucidflux": dest_root / "lucidflux" / "lucidflux.pth",
        "prompt_embeddings": dest_root / "lucidflux" / "prompt_embeddings.pt",
        "ultraflux": dest_root / "ultraflux",
        "siglip": dest_root / "siglip",
        "pid_flux_2k": dest_root / "pid" / PID_FLUX_2K_EXPERIMENT / PID_CHECKPOINT_FILE,
        "pid_flux_2kto4k": dest_root / "pid" / PID_FLUX_2KTO4K_EXPERIMENT / PID_CHECKPOINT_FILE,
        "pid_prompt_embedding": dest_root / "pid" / PID_PROMPT_EMBEDDING_FILE,
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_if_needed(src: str, dst: Path, force: bool) -> None:
    ensure_parent(dst)
    if force or not dst.exists():
        shutil.copyfile(src, dst)


def env_lines(targets: dict[str, Path]) -> tuple[str, str]:
    prefix = MODEL_KEY.replace("-", "_").upper()
    return (
        f"export {prefix}_FLOW={targets['flow']}",
        f"export {prefix}_AE={targets['ae']}",
    )


def write_env(env_path: Path, targets: dict[str, Path]) -> None:
    line1, line2 = env_lines(targets)
    env_path.write_text("\n".join([line1, line2, "", f"# source {env_path}"]) + "\n")


def write_manifest(manifest_path: Path, targets: dict[str, Path]) -> None:
    manifest = {
        "model": MODEL_KEY,
        "flow_repo": FLUX_REPO,
        "flow_file": FLOW_FILE,
        "ae_repo": FLUX_REPO,
        "ae_file": AE_FILE,
        "swinir_repo": SWINIR_REPO,
        "swinir_file": SWINIR_FILE,
        "lucidflux_repo": LUCIDFLUX_REPO,
        "lucidflux_file": LUCIDFLUX_FILE,
        "prompt_embeddings_repo": LUCIDFLUX_REPO,
        "prompt_embeddings_file": PROMPT_EMBEDDINGS_FILE,
        "pid_prompt_embedding_repo": LUCIDFLUX_REPO,
        "pid_prompt_embedding_file": PID_PROMPT_EMBEDDING_FILE,
        "ultraflux_repo": ULTRAFLUX_REPO,
        "ultraflux_subdir": "vae",
        "siglip_repo": SIGLIP_REPO,
        "pid_repo": PID_REPO,
        "pid_flux_2k_experiment": PID_FLUX_2K_EXPERIMENT,
        "pid_flux_2kto4k_experiment": PID_FLUX_2KTO4K_EXPERIMENT,
        "pid_checkpoint_file": PID_CHECKPOINT_FILE,
        "destinations": {key: str(value) for key, value in targets.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def dry_run(targets: dict[str, Path]) -> int:
    line1, line2 = env_lines(targets)
    sys.stdout.write(
        "\n".join(
            [
                f"DRY RUN: download FLOW {FLUX_REPO}:{FLOW_FILE} -> {targets['flow']}",
                f"DRY RUN: download AE {FLUX_REPO}:{AE_FILE} -> {targets['ae']}",
                f"DRY RUN: download SwinIR {SWINIR_REPO}:{SWINIR_FILE} -> {targets['swinir']}",
                f"DRY RUN: download LucidFlux {LUCIDFLUX_REPO}:{LUCIDFLUX_FILE} -> {targets['lucidflux']}",
                f"DRY RUN: download Prompt Embeddings {LUCIDFLUX_REPO}:{PROMPT_EMBEDDINGS_FILE} -> {targets['prompt_embeddings']}",
                f"DRY RUN: download PiD Prompt Embedding {LUCIDFLUX_REPO}:{PID_PROMPT_EMBEDDING_FILE} -> {targets['pid_prompt_embedding']}",
                f"DRY RUN: snapshot UltraFlux VAE {ULTRAFLUX_REPO}:vae/* -> {targets['ultraflux']}",
                f"DRY RUN: snapshot SIGLIP {SIGLIP_REPO} -> {targets['siglip']}",
                f"DRY RUN: download PiD FLUX 2k {PID_REPO}:checkpoints/{PID_FLUX_2K_EXPERIMENT}/{PID_CHECKPOINT_FILE} -> {targets['pid_flux_2k']}",
                f"DRY RUN: download PiD FLUX 2kto4k {PID_REPO}:checkpoints/{PID_FLUX_2KTO4K_EXPERIMENT}/{PID_CHECKPOINT_FILE} -> {targets['pid_flux_2kto4k']}",
                "DRY RUN: write env exports",
                line1,
                line2,
            ]
        )
        + "\n"
    )
    return 0


def main() -> int:
    args = parse_args()
    dest_root = Path(args.dest).resolve()
    targets = plan(dest_root)

    if args.dry_run:
        return dry_run(targets)

    dest_root.mkdir(parents=True, exist_ok=True)

    if args.force or not targets["flow"].exists():
        copy_if_needed(hf_hub_download(FLUX_REPO, FLOW_FILE), targets["flow"], args.force)
    if args.force or not targets["ae"].exists():
        copy_if_needed(hf_hub_download(FLUX_REPO, AE_FILE), targets["ae"], args.force)
    if args.force or not targets["swinir"].exists():
        copy_if_needed(hf_hub_download(SWINIR_REPO, SWINIR_FILE), targets["swinir"], args.force)
    if args.force or not targets["lucidflux"].exists():
        copy_if_needed(hf_hub_download(LUCIDFLUX_REPO, LUCIDFLUX_FILE), targets["lucidflux"], args.force)
    if args.force or not targets["prompt_embeddings"].exists():
        copy_if_needed(
            hf_hub_download(LUCIDFLUX_REPO, PROMPT_EMBEDDINGS_FILE),
            targets["prompt_embeddings"],
            args.force,
        )
    if args.force or not targets["pid_prompt_embedding"].exists():
        copy_if_needed(
            hf_hub_download(LUCIDFLUX_REPO, PID_PROMPT_EMBEDDING_FILE),
            targets["pid_prompt_embedding"],
            args.force,
        )
    if args.force or not targets["ultraflux"].exists():
        snapshot_download(
            ULTRAFLUX_REPO,
            allow_patterns="vae/*",
            local_dir=str(targets["ultraflux"]),
            local_dir_use_symlinks=False,
        )
    if args.force or not targets["siglip"].exists():
        snapshot_download(
            SIGLIP_REPO,
            local_dir=str(targets["siglip"]),
            local_dir_use_symlinks=False,
        )
    if args.force or not targets["pid_flux_2k"].exists():
        copy_if_needed(
            hf_hub_download(
                PID_REPO,
                f"checkpoints/{PID_FLUX_2K_EXPERIMENT}/{PID_CHECKPOINT_FILE}",
            ),
            targets["pid_flux_2k"],
            args.force,
        )
    if args.force or not targets["pid_flux_2kto4k"].exists():
        copy_if_needed(
            hf_hub_download(
                PID_REPO,
                f"checkpoints/{PID_FLUX_2KTO4K_EXPERIMENT}/{PID_CHECKPOINT_FILE}",
            ),
            targets["pid_flux_2kto4k"],
            args.force,
        )
    write_env(targets["env"], targets)
    if args.print_env:
        sys.stdout.write("\n".join(env_lines(targets)) + "\n")
    write_manifest(targets["manifest"], targets)

    sys.stdout.write("done.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
