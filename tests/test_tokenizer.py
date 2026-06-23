#!/usr/bin/env python3
"""Sample JSONL text and encode it with the local AutoTokenizer."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "pretrain_t2t_mini.jsonl"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def sample_jsonl_texts(
    data_path: Path,
    text_field: str,
    sample_size: int,
    seed: int,
) -> tuple[list[str], int]:
    rng = random.Random(seed)
    samples: list[str] = []
    seen = 0

    with data_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc

            text = record.get(text_field)
            if not isinstance(text, str) or not text:
                continue

            seen += 1
            if len(samples) < sample_size:
                samples.append(text)
                continue

            replace_at = rng.randrange(seen)
            if replace_at < sample_size:
                samples[replace_at] = text

    return samples, seen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample JSONL data and convert text to token ids with AutoTokenizer.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"JSONL data path. Default: {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Local tokenizer/model directory. Default: {DEFAULT_MODEL_DIR}",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="JSON field containing text. Default: text",
    )
    parser.add_argument(
        "--num-samples",
        type=positive_int,
        default=3,
        help="Number of records to sample. Default: 3",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable sampling. Default: 42",
    )
    parser.add_argument(
        "--max-text-chars",
        type=positive_int,
        default=160,
        help="Max text preview characters. Default: 160",
    )
    parser.add_argument(
        "--max-token-ids",
        type=positive_int,
        default=80,
        help="Max token ids to print per sample. Default: 80",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Add tokenizer special tokens during encoding. Default: false",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.data.is_file():
        print(f"data file not found: {args.data}", file=sys.stderr)
        return 1
    if not args.model_dir.is_dir():
        print(f"model directory not found: {args.model_dir}", file=sys.stderr)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        use_fast=True,
    )
    texts, total = sample_jsonl_texts(
        data_path=args.data,
        text_field=args.text_field,
        sample_size=args.num_samples,
        seed=args.seed,
    )

    if not texts:
        print(f"no usable text found in {args.data} field={args.text_field!r}", file=sys.stderr)
        return 1

    print(f"loaded tokenizer: {args.model_dir}")
    print(f"sampled {len(texts)} / {total} usable records from: {args.data}")

    for index, text in enumerate(texts, start=1):
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=args.add_special_tokens,
        )
        preview_text = text.replace("\n", "\\n")
        if len(preview_text) > args.max_text_chars:
            preview_text = f"{preview_text[: args.max_text_chars]}..."

        visible_ids = token_ids[: args.max_token_ids]
        suffix = " ..." if len(token_ids) > args.max_token_ids else ""

        print()
        print(f"[sample {index}]")
        print(f"text_chars: {len(text)}")
        print(f"token_count: {len(token_ids)}")
        print(f"text_preview: {preview_text}")
        print(f"token_ids[:{args.max_token_ids}]: {visible_ids}{suffix}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
