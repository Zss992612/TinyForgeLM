#!/usr/bin/env python3
"""Generate text from a DenseCausalLM checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ModelConfig  # noqa: E402
from src.data import load_tokenizer  # noqa: E402
from src.model.modeling_dense import DenseCausalLM  # noqa: E402


DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "tinyforgelm-stage2" / "latest.pt"
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"


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


def probability(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint path. Default: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=DEFAULT_TOKENIZER_DIR,
        help=f"Tokenizer directory. Default: {DEFAULT_TOKENIZER_DIR}",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt text. If omitted, stdin is used.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Read prompt text from this file.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=128,
        help="Maximum number of tokens to generate. Default: 128",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Use 0 for greedy decoding. Default: 0.8",
    )
    parser.add_argument(
        "--top-k",
        type=non_negative_int,
        default=50,
        help="Keep only the top-k logits before sampling. 0 disables it. Default: 50",
    )
    parser.add_argument(
        "--top-p",
        type=probability,
        default=0.95,
        help="Nucleus sampling threshold. 1 disables it. Default: 0.95",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cuda, cpu, etc. Default: auto",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Do not stop generation at eos_token_id.",
    )
    parser.add_argument(
        "--show-special-tokens",
        action="store_true",
        help="Decode special tokens instead of skipping them.",
    )
    return parser.parse_args()


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8")
    if args.prompt is not None:
        return args.prompt
    return sys.stdin.read()


def load_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[DenseCausalLM, ModelConfig, dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = DenseCausalLM(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, model_config, checkpoint


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    values, _ = torch.topk(logits, top_k)
    cutoff = values[..., -1, None]
    return logits.masked_fill(logits < cutoff, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p <= 0.0 or top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = sorted_probs.cumsum(dim=-1)

    sorted_remove_mask = cumulative_probs > top_p
    sorted_remove_mask[..., 1:] = sorted_remove_mask[..., :-1].clone()
    sorted_remove_mask[..., 0] = False

    remove_mask = torch.zeros_like(sorted_remove_mask)
    remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_remove_mask)
    return logits.masked_fill(remove_mask, float("-inf"))


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: DenseCausalLM,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_token_id: int | None,
    stop_at_eos: bool,
) -> torch.Tensor:
    generated = input_ids
    context_length = model.config.max_position_embeddings

    for _ in range(max_new_tokens):
        model_input = generated[:, -context_length:]
        logits = model(model_input)
        next_token_logits = logits[:, -1, :]
        next_token = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated = torch.cat([generated, next_token], dim=-1)

        if stop_at_eos and eos_token_id is not None:
            if int(next_token.item()) == eos_token_id:
                break

    return generated


def main() -> int:
    args = parse_args()

    try:
        torch.manual_seed(args.seed)
        device = select_device(args.device)
        tokenizer = load_tokenizer(args.tokenizer_dir)
        model, model_config, checkpoint = load_model(args.checkpoint, device=device)

        prompt = read_prompt(args)
        if not prompt:
            raise ValueError("prompt is empty")

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids:
            raise ValueError("prompt produced no token ids")

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated_ids = generate(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            stop_at_eos=not args.ignore_eos,
        )

        output_ids = generated_ids[0].tolist()
        text = tokenizer.decode(
            output_ids,
            skip_special_tokens=not args.show_special_tokens,
        )

        print(f"checkpoint_step: {checkpoint.get('step')}")
        print(f"device: {device}")
        print(f"context_length: {model_config.max_position_embeddings}")
        print(f"prompt_tokens: {len(prompt_ids)}")
        print(f"total_tokens: {len(output_ids)}")
        print()
        print(text)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
