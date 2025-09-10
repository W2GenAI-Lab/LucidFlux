import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict

from huggingface_hub import hf_hub_download, snapshot_download


# Fixed sources and filenames (no customization per user request)
FLUX_REPO = "black-forest-labs/FLUX.1-dev"
FLOW_FILE = "flux1-dev.safetensors"
AE_FILE = "ae.safetensors"
SWINIR_REPO = "lxq007/DiffBIR"
SWINIR_FILE = "general_swinir_v1.ckpt"
T5_REPO = "XLabs-AI/xflux_text_encoders"
CLIP_REPO = "openai/clip-vit-large-patch14"
LUCIDFLUX_REPO = "W2GenAI/LucidFlux"
LUCIDFLUX_FILE = "lucidflux.pth"
SIGLIP_REPO = "google/siglip2-so400m-patch16-512"
MODEL_KEY = "flux-dev"  # prefix for env vars


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download fixed weights (flux-dev + SwinIR) and emit env exports")
    p.add_argument("--dest", type=str, default="weights", help="Destination root directory")
    p.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    p.add_argument("--print-env", action="store_true", help="Also print export lines to stdout")
    p.add_argument("--force", action="store_true", help="Overwrite existing files")
    return p.parse_args()


def ensure_dir(p: Path) -> None:
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)


def plan(dest_root: Path) -> Tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    model_dir = dest_root / MODEL_KEY
    flow_dst = model_dir / FLOW_FILE
    ae_dst = model_dir / AE_FILE
    env_path = dest_root / "env.sh"
    manifest_path = dest_root / "manifest.json"
    swinir_dst = dest_root / "swinir.pth"
    t5_dir = dest_root / "t5"
    clip_dir = dest_root / "clip"
    lucidflux_dst = dest_root / "lucidflux" / "lucidflux.pth"
    siglip_dir = dest_root / "siglip"
    return flow_dst, ae_dst, env_path, manifest_path, swinir_dst, t5_dir, clip_dir, lucidflux_dst, siglip_dir


def env_lines(flow_dst: Path, ae_dst: Path, t5_dir: Path, clip_dir: Path) -> Tuple[str, str, str, str]:
    prefix = MODEL_KEY.replace('-', '_').upper()
    return (
        f"export {prefix}_FLOW={flow_dst}",
        f"export {prefix}_AE={ae_dst}",
        f"export T5_PATH={t5_dir}",
        f"export CLIP_PATH={clip_dir}",
    )


def write_env(env_path: Path, flow_dst: Path, ae_dst: Path, t5_dir: Path, clip_dir: Path) -> None:
    l1, l2, l3, l4 = env_lines(flow_dst, ae_dst, t5_dir, clip_dir)
    content = "\n".join([l1, l2, l3, l4, "", f"# source {env_path}"]) + "\n"
    env_path.write_text(content)


def write_manifest(path: Path, data: Dict[str, str]) -> None:
    import json
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    dest_root = Path(args.dest).resolve()

    flow_dst, ae_dst, env_path, manifest_path, swinir_dst, t5_dir, clip_dir, lucidflux_dst, siglip_dir = plan(dest_root)

    if args.dry_run:
        l1, l2, l3, l4 = env_lines(flow_dst, ae_dst, t5_dir, clip_dir)
        sys.stdout.write(
            "\n".join(
                [
                    f"DRY RUN: download FLOW {FLUX_REPO}:{FLOW_FILE} -> {flow_dst}",
                    f"DRY RUN: download AE   {FLUX_REPO}:{AE_FILE} -> {ae_dst}",
                    f"DRY RUN: download SwinIR {SWINIR_REPO}:{SWINIR_FILE} -> {swinir_dst}",
                    f"DRY RUN: snapshot T5   {T5_REPO} -> {t5_dir}",
                    f"DRY RUN: snapshot CLIP {CLIP_REPO} -> {clip_dir}",
                    f"DRY RUN: download LucidFlux {LUCIDFLUX_REPO}:{LUCIDFLUX_FILE} -> {lucidflux_dst}",
                    f"DRY RUN: snapshot SIGLIP {SIGLIP_REPO} -> {siglip_dir}",
                    "DRY RUN: write env exports",
                    l1,
                    l2,
                    l3,
                    l4,
                ]
            )
            + "\n"
        )
        return 0

    ensure_dir(dest_root)
    ensure_dir(flow_dst.parent)
    ensure_dir(lucidflux_dst.parent)

    # Download via HF cache then copy into dest
    flow_src = hf_hub_download(FLUX_REPO, FLOW_FILE)
    ae_src = hf_hub_download(FLUX_REPO, AE_FILE)
    swinir_src = hf_hub_download(SWINIR_REPO, SWINIR_FILE)
    # Full snapshots for local T5/CLIP (no symlinks to keep folder portable)
    snapshot_download(T5_REPO, local_dir=str(t5_dir), local_dir_use_symlinks=False)
    snapshot_download(CLIP_REPO, local_dir=str(clip_dir), local_dir_use_symlinks=False)
    snapshot_download(SIGLIP_REPO, local_dir=str(siglip_dir), local_dir_use_symlinks=False)
    lucidflux_src = hf_hub_download(LUCIDFLUX_REPO, LUCIDFLUX_FILE)

    if args.force or not flow_dst.exists():
        flow_dst.write_bytes(Path(flow_src).read_bytes())
    if args.force or not ae_dst.exists():
        ae_dst.write_bytes(Path(ae_src).read_bytes())
    if args.force or not swinir_dst.exists():
        swinir_dst.write_bytes(Path(swinir_src).read_bytes())
    if args.force or not lucidflux_dst.exists():
        lucidflux_dst.write_bytes(Path(lucidflux_src).read_bytes())

    write_env(env_path, flow_dst, ae_dst, t5_dir, clip_dir)
    if args.print_env:
        l1, l2, l3, l4 = env_lines(flow_dst, ae_dst, t5_dir, clip_dir)
        sys.stdout.write("\n".join([l1, l2, l3, l4]) + "\n")

    write_manifest(
        manifest_path,
        {
            "model": MODEL_KEY,
            "flow_repo": FLUX_REPO,
            "flow_file": FLOW_FILE,
            "ae_repo": FLUX_REPO,
            "ae_file": AE_FILE,
            "swinir_repo": SWINIR_REPO,
            "swinir_file": SWINIR_FILE,
            "t5_repo": T5_REPO,
            "clip_repo": CLIP_REPO,
            "lucidflux_repo": LUCIDFLUX_REPO,
            "lucidflux_file": LUCIDFLUX_FILE,
            "siglip_repo": SIGLIP_REPO,
        },
    )

    sys.stdout.write("done.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
