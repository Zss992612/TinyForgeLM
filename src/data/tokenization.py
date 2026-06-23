"""Tokenization helpers for JSONL text data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_tokenizer(
    tokenizer_dir: str | Path,
    *,
    use_fast: bool = True,
) -> PreTrainedTokenizerBase:
    path = Path(tokenizer_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"tokenizer directory not found: {path}")

    return AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        use_fast=use_fast,
    )


def tokenize_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    add_special_tokens: bool = False,
    append_eos: bool = True,
    max_length: int | None = None,
) -> list[int]:
    if max_length is not None and max_length <= 0:
        raise ValueError("max_length must be a positive integer")

    encode_kwargs: dict[str, Any] = {
        "add_special_tokens": add_special_tokens,
    }

    if max_length is not None and not (append_eos and max_length == 1):
        encode_kwargs["max_length"] = max_length - 1 if append_eos else max_length
        encode_kwargs["truncation"] = True

    input_ids = [] if append_eos and max_length == 1 else tokenizer.encode(
        text,
        **encode_kwargs,
    )

    if append_eos:
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("tokenizer does not define eos_token_id")
        if not input_ids or input_ids[-1] != eos_token_id:
            input_ids.append(eos_token_id)

    return input_ids


def tokenize_jsonl_file(
    input_path: str | Path,
    output_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    *,
    text_field: str = "text",
    add_special_tokens: bool = False,
    append_eos: bool = True,
    max_length: int | None = None,
    keep_text: bool = False,
    limit: int | None = None,
) -> int:
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.is_file():
        raise FileNotFoundError(f"input file not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with input_file.open("r", encoding="utf-8") as reader, output_file.open(
        "w",
        encoding="utf-8",
    ) as writer:
        for line_number, line in enumerate(reader, start=1):
            if limit is not None and written >= limit:
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
            output_record: dict[str, Any] = {
                "input_ids": input_ids,
                "token_count": len(input_ids),
            }
            if keep_text:
                output_record["text"] = text

            writer.write(json.dumps(output_record, ensure_ascii=False))
            writer.write("\n")
            written += 1

    return written
