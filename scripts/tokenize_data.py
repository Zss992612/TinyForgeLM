#!/usr/bin/env python3
"""Convert JSONL text data into token id JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_tokenizer, tokenize_jsonl_file  # noqa: E402


DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "pretrain_t2t_test_100.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "pretrain_t2t_test_100_tokenized.jsonl"
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize JSONL text data with the local AutoTokenizer.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input JSONL path. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSONL path. Default: {DEFAULT_OUTPUT_PATH}",
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
        help="Truncate token ids to this length. Default: no truncation",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Maximum records to tokenize. Default: all records",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Add tokenizer special tokens during encoding. Default: false",
    )
    parser.add_argument(
        "--no-eos",
        action="store_true",
        help="Do not append eos_token_id to each tokenized record. Default: append EOS",
    )
    parser.add_argument(
        "--keep-text",
        action="store_true",
        help="Keep raw text in the output JSONL. Default: false",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        tokenizer = load_tokenizer(args.tokenizer_dir)
        count = tokenize_jsonl_file(
            input_path=args.input,
            output_path=args.output,
            tokenizer=tokenizer,
            text_field=args.text_field,
            add_special_tokens=args.add_special_tokens,
            append_eos=not args.no_eos,
            max_length=args.max_length,
            keep_text=args.keep_text,
            limit=args.limit,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"loaded tokenizer: {args.tokenizer_dir}")
    print(f"tokenized records: {count}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
