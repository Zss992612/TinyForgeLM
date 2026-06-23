"""Build binary token files for efficient training data loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from transformers import PreTrainedTokenizerBase

from .tokenization import tokenize_text


def _as_unsigned_integer_dtype(dtype: str | np.dtype[Any]) -> np.dtype[Any]:
    np_dtype = np.dtype(dtype)
    if not np.issubdtype(np_dtype, np.unsignedinteger):
        raise ValueError(f"dtype must be an unsigned integer dtype, got {np_dtype}")
    return np_dtype


def _validate_token_ids(
    input_ids: list[int],
    *,
    dtype: np.dtype[Any],
    line_number: int,
) -> None:
    if not input_ids:
        raise ValueError(f"line {line_number} produced no token ids")

    dtype_max = int(np.iinfo(dtype).max)
    max_token_id = max(input_ids)
    min_token_id = min(input_ids)

    if min_token_id < 0:
        raise ValueError(f"line {line_number} contains negative token id {min_token_id}")
    if max_token_id > dtype_max:
        raise ValueError(
            f"line {line_number} token id {max_token_id} exceeds {dtype} max {dtype_max}"
        )


def build_token_bin_from_jsonl(
    input_path: str | Path,
    output_bin_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    *,
    output_meta_path: str | Path | None = None,
    text_field: str = "text",
    add_special_tokens: bool = False,
    append_eos: bool = True,
    max_length: int | None = None,
    dtype: str | np.dtype[Any] = "uint16",
    limit: int | None = None,
) -> dict[str, Any]:
    """Tokenize raw JSONL text and stream token ids into a binary file."""

    input_file = Path(input_path)
    output_bin_file = Path(output_bin_path)
    output_meta_file = (
        Path(output_meta_path)
        if output_meta_path is not None
        else output_bin_file.with_suffix(".meta.json")
    )
    np_dtype = _as_unsigned_integer_dtype(dtype)

    if not input_file.is_file():
        raise FileNotFoundError(f"input file not found: {input_file}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")

    output_bin_file.parent.mkdir(parents=True, exist_ok=True)
    output_meta_file.parent.mkdir(parents=True, exist_ok=True)

    num_records = 0
    num_tokens = 0
    max_record_tokens = 0
    min_record_tokens: int | None = None
    max_token_id = -1

    with input_file.open("r", encoding="utf-8") as reader, output_bin_file.open(
        "wb"
    ) as writer:
        for line_number, line in enumerate(reader, start=1):
            if limit is not None and num_records >= limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc

            text = record.get(text_field)
            if not isinstance(text, str) or not text:
                raise ValueError(
                    f"line {line_number} missing non-empty text field {text_field!r}"
                )

            input_ids = tokenize_text(
                text,
                tokenizer,
                add_special_tokens=add_special_tokens,
                append_eos=append_eos,
                max_length=max_length,
            )
            _validate_token_ids(input_ids, dtype=np_dtype, line_number=line_number)

            token_array = np.asarray(input_ids, dtype=np_dtype)
            token_array.tofile(writer)

            record_tokens = len(input_ids)
            num_records += 1
            num_tokens += record_tokens
            max_record_tokens = max(max_record_tokens, record_tokens)
            min_record_tokens = (
                record_tokens
                if min_record_tokens is None
                else min(min_record_tokens, record_tokens)
            )
            max_token_id = max(max_token_id, int(token_array.max()))

    if num_records == 0:
        raise ValueError(f"no usable records found in {input_file}")

    metadata: dict[str, Any] = {
        "input_path": str(input_file),
        "output_bin_path": str(output_bin_file),
        "output_meta_path": str(output_meta_file),
        "dtype": str(np_dtype),
        "num_records": num_records,
        "num_tokens": num_tokens,
        "min_record_tokens": min_record_tokens,
        "max_record_tokens": max_record_tokens,
        "max_token_id": max_token_id,
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "text_field": text_field,
        "add_special_tokens": add_special_tokens,
        "append_eos": append_eos,
        "max_length": max_length,
    }

    output_meta_file.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metadata
