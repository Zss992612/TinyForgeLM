"""Datasets backed by binary token files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def load_token_bin_metadata(meta_path: str | Path) -> dict[str, Any]:
    path = Path(meta_path)
    if not path.is_file():
        raise FileNotFoundError(f"metadata file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


class TokenBinDataset(Dataset[dict[str, torch.Tensor]]):
    """Map-style dataset that slices fixed-length blocks from a token memmap."""

    def __init__(
        self,
        bin_path: str | Path,
        *,
        block_size: int,
        dtype: str | np.dtype[Any] | None = None,
        meta_path: str | Path | None = None,
        stride: int | None = None,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be a positive integer")

        self.bin_path = Path(bin_path)
        if not self.bin_path.is_file():
            raise FileNotFoundError(f"token bin file not found: {self.bin_path}")

        self.metadata: dict[str, Any] | None = None
        if meta_path is not None:
            self.metadata = load_token_bin_metadata(meta_path)
            if dtype is None:
                dtype = self.metadata.get("dtype")

        if dtype is None:
            dtype = "uint16"

        self.dtype = np.dtype(dtype)
        self.block_size = block_size
        self.stride = block_size if stride is None else stride
        if self.stride <= 0:
            raise ValueError("stride must be a positive integer")

        self.tokens = np.memmap(self.bin_path, dtype=self.dtype, mode="r")
        self.num_tokens = int(self.tokens.shape[0])
        if self.metadata is not None and "num_tokens" in self.metadata:
            expected_num_tokens = int(self.metadata["num_tokens"])
            if self.num_tokens != expected_num_tokens:
                raise ValueError(
                    f"metadata num_tokens={expected_num_tokens} does not match "
                    f"bin tokens={self.num_tokens}"
                )
        if self.num_tokens < self.block_size:
            raise ValueError(
                f"not enough tokens for one block: num_tokens={self.num_tokens}, "
                f"block_size={self.block_size}"
            )

    def __len__(self) -> int:
        return ((self.num_tokens - self.block_size) // self.stride) + 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        start = index * self.stride
        end = start + self.block_size
        token_block = np.asarray(self.tokens[start:end], dtype=np.int64)
        input_ids = torch.from_numpy(token_block)

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }
