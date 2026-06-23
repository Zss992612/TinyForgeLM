#!/usr/bin/env python3
"""Download MiniMind JSONL files from ModelScope.

This script only downloads raw JSONL files. Run scripts/prepare_train_data.py
afterward to convert the text data into the binary token format used for
training.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ID = "gongjy/minimind_dataset"
DEFAULT_LOCAL_DIR = PROJECT_ROOT / "data"
DEFAULT_FILES = ("pretrain_t2t.jsonl",)
SFT_FILES = ("sft_t2t.jsonl",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MiniMind dataset files from ModelScope.",
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=f"ModelScope dataset id. Default: {DEFAULT_DATASET_ID}",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR,
        help=f"Download directory. Default: {DEFAULT_LOCAL_DIR}",
    )
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        default=None,
        help=(
            "Dataset file to download. Can be passed multiple times. "
            "Default: pretrain_t2t.jsonl"
        ),
    )
    parser.add_argument(
        "--sft",
        action="store_true",
        help="Also download the SFT JSONL file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print ModelScope commands without running them.",
    )
    return parser.parse_args()


def resolve_local_dir(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_files(args: argparse.Namespace) -> list[str]:
    files = list(args.files) if args.files else list(DEFAULT_FILES)
    if args.sft:
        files.extend(SFT_FILES)
    return list(dict.fromkeys(files))


def require_modelscope_cli() -> str:
    executable = shutil.which("modelscope")
    if executable is None:
        raise FileNotFoundError(
            "ModelScope CLI not found. Install it first, for example:\n"
            "  uv pip install modelscope\n"
            "or:\n"
            "  pip install -U modelscope"
        )
    return executable


def download_file(
    *,
    modelscope_cli: str,
    dataset_id: str,
    filename: str,
    local_dir: Path,
    dry_run: bool,
) -> None:
    command = [
        modelscope_cli,
        "download",
        "--dataset",
        dataset_id,
        filename,
        "--local_dir",
        str(local_dir),
    ]
    print(" ".join(command))
    if dry_run:
        return

    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    local_dir = resolve_local_dir(args.local_dir)
    files = get_files(args)

    try:
        modelscope_cli = "modelscope" if args.dry_run else require_modelscope_cli()
        local_dir.mkdir(parents=True, exist_ok=True)

        for filename in files:
            download_file(
                modelscope_cli=modelscope_cli,
                dataset_id=args.dataset_id,
                filename=filename,
                local_dir=local_dir,
                dry_run=args.dry_run,
            )

    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print("download finished")
    print(f"local_dir: {local_dir}")
    print("next: uv run python scripts/prepare_train_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
