import os
import sys
import argparse
import subprocess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Login to Hugging Face using a token.")
    p.add_argument("--token", type=str, default=None, help="HF token; falls back to $HF_TOKEN")
    p.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    token = args.token if args.token is not None else os.environ.get("HF_TOKEN")

    if token is None or len(token) == 0:
        sys.stderr.write("error: missing Hugging Face token. Provide --token or set $HF_TOKEN.\n")
        return 2

    if args.dry_run:
        sys.stdout.write("DRY RUN: would execute 'huggingface-cli login --token ***REDACTED*** --add-to-git-credential'\n")
        return 0

    cmd = [
        sys.executable,
        "-m",
        "huggingface_hub.commands.huggingface_cli",
        "login",
        "--token",
        token,
        "--add-to-git-credential",
    ]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        sys.stderr.write("error: huggingface-cli login failed.\n")
        return proc.returncode

    sys.stdout.write("login: success.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

