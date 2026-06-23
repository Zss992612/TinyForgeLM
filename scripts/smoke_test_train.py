#!/usr/bin/env python3
"""Overfit a tiny DenseCausalLM on a small batch of real tokenized text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.optim import AdamW


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ModelConfig  # noqa: E402
from src.model.modeling_dense import DenseCausalLM  # noqa: E402
from src.training.losses import causal_lm_loss  # noqa: E402


DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "pretrain_t2t_test_100_tokenized.jsonl"
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer.json"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit a tiny DenseCausalLM on a fixed real-text token batch.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Tokenized JSONL path. Default: {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=4,
        help="Fixed overfit batch size. Default: 4",
    )
    parser.add_argument(
        "--seq-len",
        type=positive_int,
        default=64,
        help="Sequence length per sample. Default: 64",
    )
    parser.add_argument(
        "--steps",
        type=positive_int,
        default=120,
        help="Number of optimization steps. Default: 120",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-3,
        help="AdamW learning rate. Default: 3e-3",
    )
    parser.add_argument(
        "--tokenizer-json",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help=f"Tokenizer JSON used to read vocab size. Default: {DEFAULT_TOKENIZER_PATH}",
    )
    return parser.parse_args()


def build_tiny_config(vocab_size: int, max_position_embeddings: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=True,
    )


def load_token_stream(data_path: Path) -> list[int]:
    if not data_path.is_file():
        raise FileNotFoundError(f"tokenized data file not found: {data_path}")

    token_ids: list[int] = []
    with data_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            input_ids = record.get("input_ids")
            if not isinstance(input_ids, list) or not input_ids:
                raise ValueError(f"line {line_number} missing non-empty input_ids")
            if not all(isinstance(token_id, int) for token_id in input_ids):
                raise ValueError(f"line {line_number} contains non-integer token ids")

            token_ids.extend(input_ids)

    if not token_ids:
        raise ValueError(f"no token ids loaded from {data_path}")

    return token_ids


def load_vocab_size(tokenizer_json_path: Path, token_ids: list[int]) -> int:
    if tokenizer_json_path.is_file():
        tokenizer_config = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
        vocab = tokenizer_config.get("model", {}).get("vocab")
        if isinstance(vocab, dict):
            return len(vocab)

    return max(token_ids) + 1


def make_fixed_real_batch(
    token_ids: list[int],
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    required_tokens = batch_size * seq_len
    if len(token_ids) < required_tokens:
        raise ValueError(
            f"need at least {required_tokens} tokens, got {len(token_ids)}"
        )

    batch_tokens = token_ids[:required_tokens]
    return torch.tensor(
        batch_tokens,
        device=device,
        dtype=torch.long,
    ).view(batch_size, seq_len)


def main() -> int:
    args = parse_args()
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_ids = load_token_stream(args.data)
    vocab_size = load_vocab_size(args.tokenizer_json, token_ids)
    max_token_id = max(token_ids)
    if max_token_id >= vocab_size:
        raise ValueError(
            f"token id {max_token_id} is outside vocab_size={vocab_size}"
        )

    config = build_tiny_config(
        vocab_size=vocab_size,
        max_position_embeddings=args.seq_len,
    )
    model = DenseCausalLM(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    input_ids = make_fixed_real_batch(
        token_ids,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
    )

    print(f"device={device}")
    print(f"data={args.data}")
    print(f"loaded_tokens={len(token_ids)}")
    print(f"vocab_size={config.vocab_size}")
    print(f"max_token_id={max_token_id}")
    print(f"batch_shape={tuple(input_ids.shape)}")

    losses: list[float] = []
    model.train()

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        logits = model(input_ids)
        loss = causal_lm_loss(logits=logits, labels=input_ids)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)

        if step == 1 or step % 10 == 0 or step == args.steps:
            print(f"step={step:03d} loss={loss_value:.4f}")

    initial_loss = losses[0]
    final_loss = losses[-1]
    print(f"initial_loss={initial_loss:.4f}")
    print(f"final_loss={final_loss:.4f}")

    if final_loss >= initial_loss:
        raise RuntimeError(
            f"loss did not decrease: initial={initial_loss:.4f}, "
            f"final={final_loss:.4f}"
        )

    print("real-text overfit smoke train passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
