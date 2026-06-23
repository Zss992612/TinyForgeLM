#!/usr/bin/env python3
"""Prepare training token data and verify DataLoader batches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    TokenBinDataset,
    build_token_bin_from_jsonl,
    load_token_bin_metadata,
    load_tokenizer,
)


DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "pretrain_t2t_mini.jsonl"
DEFAULT_OUTPUT_BIN_PATH = PROJECT_ROOT / "data" / "pretrain_t2t_mini.tokens.bin"
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build tokens.bin/meta.json from raw JSONL text, then verify loading "
            "fixed-length training batches with DataLoader."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML data config path. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"Input raw JSONL path. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output-bin",
        type=Path,
        default=None,
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
        default=None,
        help=f"Local tokenizer directory. Default: {DEFAULT_TOKENIZER_DIR}",
    )
    parser.add_argument(
        "--text-field",
        default=None,
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
        help="Maximum records to preprocess. Default: all records",
    )
    parser.add_argument(
        "--dtype",
        default=None,
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
    parser.add_argument(
        "--reuse-bin",
        action="store_true",
        help="Skip preprocessing and load an existing output bin/meta pair.",
    )
    parser.add_argument(
        "--block-size",
        type=positive_int,
        default=None,
        help="Sequence length per training sample. Default: 2048",
    )
    parser.add_argument(
        "--stride",
        type=positive_int,
        default=None,
        help="Distance between sample starts. Default: block-size",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=None,
        help="DataLoader batch size. Default: 4",
    )
    parser.add_argument(
        "--num-workers",
        type=non_negative_int,
        default=None,
        help="DataLoader worker count. Default: 0",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable pinned memory for CUDA training loaders.",
    )
    parser.add_argument(
        "--drop-last",
        action="store_true",
        help="Drop the last incomplete batch.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable DataLoader shuffle. Default: shuffle training blocks",
    )
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict[str, object]:
    if not config_path.is_file():
        raise FileNotFoundError(f"data config file not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"data config must be a mapping: {config_path}")
    return config


def _get_path(
    cli_value: Path | None,
    config: dict[str, object],
    key: str,
    default: Path | None,
) -> Path | None:
    if cli_value is not None:
        return cli_value

    value = config.get(key, default)
    if value is None:
        return None

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _get_value(
    cli_value: object,
    config: dict[str, object],
    key: str,
    default: object,
) -> object:
    return cli_value if cli_value is not None else config.get(key, default)


def _get_flag(
    enabled_by_cli: bool,
    config: dict[str, object],
    key: str,
    default: bool,
) -> bool:
    return enabled_by_cli or bool(config.get(key, default))


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    config = load_yaml_config(args.config)

    args.input = _get_path(args.input, config, "input", DEFAULT_INPUT_PATH)
    args.output_bin = _get_path(
        args.output_bin,
        config,
        "output_bin",
        DEFAULT_OUTPUT_BIN_PATH,
    )
    args.output_meta = _get_path(args.output_meta, config, "output_meta", None)
    args.tokenizer_dir = _get_path(
        args.tokenizer_dir,
        config,
        "tokenizer_dir",
        DEFAULT_TOKENIZER_DIR,
    )

    args.text_field = _get_value(args.text_field, config, "text_field", "text")
    args.max_length = _get_value(args.max_length, config, "max_length", None)
    args.limit = _get_value(args.limit, config, "limit", None)
    args.dtype = _get_value(args.dtype, config, "dtype", "uint16")
    args.block_size = _get_value(args.block_size, config, "block_size", 2048)
    args.stride = _get_value(args.stride, config, "stride", None)
    args.batch_size = _get_value(args.batch_size, config, "batch_size", 4)
    args.num_workers = _get_value(args.num_workers, config, "num_workers", 0)

    args.add_special_tokens = _get_flag(
        args.add_special_tokens,
        config,
        "add_special_tokens",
        False,
    )
    append_eos = bool(config.get("append_eos", True))
    args.append_eos = False if args.no_eos else append_eos
    args.reuse_bin = _get_flag(args.reuse_bin, config, "reuse_bin", False)
    args.pin_memory = _get_flag(args.pin_memory, config, "pin_memory", False)
    args.drop_last = _get_flag(args.drop_last, config, "drop_last", False)
    shuffle = bool(config.get("shuffle", True))
    args.shuffle = False if args.no_shuffle else shuffle

    return args


def resolve_meta_path(output_bin: Path, output_meta: Path | None) -> Path:
    return output_meta if output_meta is not None else output_bin.with_suffix(".meta.json")


def build_or_load_metadata(args: argparse.Namespace) -> dict[str, object]:
    meta_path = resolve_meta_path(args.output_bin, args.output_meta)

    if args.reuse_bin:
        return load_token_bin_metadata(meta_path)

    tokenizer = load_tokenizer(args.tokenizer_dir)
    return build_token_bin_from_jsonl(
        input_path=args.input,
        output_bin_path=args.output_bin,
        output_meta_path=meta_path,
        tokenizer=tokenizer,
        text_field=args.text_field,
        add_special_tokens=args.add_special_tokens,
        append_eos=args.append_eos,
        max_length=args.max_length,
        dtype=args.dtype,
        limit=args.limit,
    )


def build_dataloader(args: argparse.Namespace) -> tuple[TokenBinDataset, DataLoader]:
    meta_path = resolve_meta_path(args.output_bin, args.output_meta)
    dataset = TokenBinDataset(
        args.output_bin,
        meta_path=meta_path,
        block_size=args.block_size,
        stride=args.stride,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )
    return dataset, dataloader


def main() -> int:
    args = resolve_args(parse_args())

    try:
        metadata = build_or_load_metadata(args)
        dataset, dataloader = build_dataloader(args)
        batch = next(iter(dataloader))
    except (FileNotFoundError, ValueError, StopIteration) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"input: {metadata.get('input_path', args.input)}")
    print(f"output_bin: {metadata.get('output_bin_path', args.output_bin)}")
    print(f"output_meta: {metadata.get('output_meta_path', resolve_meta_path(args.output_bin, args.output_meta))}")
    print(f"records: {metadata.get('num_records')}")
    print(f"tokens: {metadata.get('num_tokens')}")
    print(f"dtype: {metadata.get('dtype')}")
    print(f"eos_token_id: {metadata.get('eos_token_id')}")
    print(f"dataset_len: {len(dataset)}")
    print(f"block_size: {args.block_size}")
    print(f"batch_size: {args.batch_size}")
    print(f"input_ids_shape: {tuple(batch['input_ids'].shape)}")
    print(f"labels_shape: {tuple(batch['labels'].shape)}")
    print(f"input_ids_dtype: {batch['input_ids'].dtype}")
    print(f"labels_equal_input_ids: {bool((batch['labels'] == batch['input_ids']).all())}")
    print("data pipeline ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
