#!/usr/bin/env python3
"""Build a binary token stream from raw JSONL text data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_token_bin_from_jsonl, load_tokenizer  # noqa: E402


DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "pretrain_t2t_mini.jsonl"
DEFAULT_OUTPUT_BIN_PATH = PROJECT_ROOT / "data" / "pretrain_t2t_mini.tokens.bin"
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize raw JSONL text and write a contiguous token bin file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input raw JSONL path. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output-bin",
        type=Path,
        default=DEFAULT_OUTPUT_BIN_PATH,
        help=f"Output token bin path. Default: {DEFAULT_OUTPUT_BIN_PATH}",
    )
    parser.add_argument(
        "--output-meta",
        type=Path,
        default=None,
        help="Output metadata JSON path. Default: output bin with .meta.json suffix",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=DEFAULT_TOKENIZER_DIR,
        help=f"Local tokenizer directory. Default: {DEFAULT_TOKENIZER_DIR}",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="JSON field containing raw text. Default: text",
    )
    parser.add_argument(
        "--max-length",
        type=positive_int,
        default=None,
        help="Truncate each record to this length including EOS. Default: no truncation",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Maximum records to process. Default: all records",
    )
    parser.add_argument(
        "--dtype",
        default="uint16",
        help="Unsigned integer dtype for token ids. Default: uint16",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Ask tokenizer to add special tokens before appending EOS. Default: false",
    )
    parser.add_argument(
        "--no-eos",
        action="store_true",
        help="Do not append eos_token_id to each record. Default: append EOS",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        tokenizer = load_tokenizer(args.tokenizer_dir)
        metadata = build_token_bin_from_jsonl(
            input_path=args.input,
            output_bin_path=args.output_bin,
            output_meta_path=args.output_meta,
            tokenizer=tokenizer,
            text_field=args.text_field,
            add_special_tokens=args.add_special_tokens,
            append_eos=not args.no_eos,
            max_length=args.max_length,
            dtype=args.dtype,
            limit=args.limit,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"loaded tokenizer: {args.tokenizer_dir}")
    print(f"input: {args.input}")
    print(f"output_bin: {metadata['output_bin_path']}")
    print(f"records: {metadata['num_records']}")
    print(f"tokens: {metadata['num_tokens']}")
    print(f"dtype: {metadata['dtype']}")
    print(f"eos_token_id: {metadata['eos_token_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
