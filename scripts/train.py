#!/usr/bin/env python3
"""Train DenseCausalLM on a prepared token bin dataset."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ModelConfig  # noqa: E402
from src.data import TokenBinDataset, load_token_bin_metadata  # noqa: E402
from src.model.modeling_dense import DenseCausalLM  # noqa: E402
from src.training.losses import causal_lm_loss  # noqa: E402


DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"
DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train.yaml"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyForgeLM DenseCausalLM.")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help=f"Model YAML config path. Default: {DEFAULT_MODEL_CONFIG}",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help=f"Data YAML config path. Default: {DEFAULT_DATA_CONFIG}",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=DEFAULT_TRAIN_CONFIG,
        help=f"Train YAML config path. Default: {DEFAULT_TRAIN_CONFIG}",
    )
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=None,
        help="Override train max_steps.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Override checkpoint path to resume from.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def resolve_project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_meta_path(output_bin: Path, output_meta: str | Path | None) -> Path:
    meta_path = resolve_project_path(output_meta)
    return meta_path if meta_path is not None else output_bin.with_suffix(".meta.json")


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(data_config: dict[str, Any]) -> tuple[TokenBinDataset, DataLoader]:
    output_bin = resolve_project_path(data_config.get("output_bin"))
    if output_bin is None:
        raise ValueError("data config must define output_bin")

    output_meta = resolve_meta_path(output_bin, data_config.get("output_meta"))
    if not output_bin.is_file() or not output_meta.is_file():
        raise FileNotFoundError(
            "prepared token bin/meta not found; run scripts/prepare_train_data.py first"
        )

    block_size = int(data_config.get("block_size", 2048))
    batch_size = int(data_config.get("batch_size", 4))
    num_workers = int(data_config.get("num_workers", 0))

    dataset = TokenBinDataset(
        output_bin,
        meta_path=output_meta,
        block_size=block_size,
        stride=data_config.get("stride"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(data_config.get("shuffle", True)),
        num_workers=num_workers,
        pin_memory=bool(data_config.get("pin_memory", False)),
        drop_last=bool(data_config.get("drop_last", False)),
    )
    return dataset, dataloader


def check_config_compatibility(
    model_config: ModelConfig,
    data_config: dict[str, Any],
    dataset: TokenBinDataset,
) -> None:
    if dataset.block_size > model_config.max_position_embeddings:
        raise ValueError(
            f"block_size={dataset.block_size} exceeds "
            f"max_position_embeddings={model_config.max_position_embeddings}"
        )

    metadata = dataset.metadata or {}
    max_token_id = metadata.get("max_token_id")
    if max_token_id is not None and int(max_token_id) >= model_config.vocab_size:
        raise ValueError(
            f"data max_token_id={max_token_id} exceeds model vocab_size="
            f"{model_config.vocab_size}"
        )

    data_batch_size = int(data_config.get("batch_size", 4))
    if data_batch_size <= 0:
        raise ValueError("batch_size must be positive")


def cycle_batches(dataloader: DataLoader):
    while True:
        yielded = False
        for batch in dataloader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("dataloader produced no batches")


def get_learning_rate(step: int, train_config: dict[str, Any]) -> float:
    base_lr = float(train_config.get("learning_rate", 3e-4))
    min_lr = float(train_config.get("min_learning_rate", 0.0))
    warmup_steps = int(train_config.get("warmup_steps", 0))
    max_steps = int(train_config.get("max_steps", 1000))
    lr_decay = bool(train_config.get("lr_decay", True))

    if base_lr <= 0:
        raise ValueError("learning_rate must be positive")
    if min_lr < 0:
        raise ValueError("min_learning_rate must be non-negative")
    if min_lr > base_lr:
        raise ValueError("min_learning_rate must be <= learning_rate")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")

    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps

    if not lr_decay:
        return base_lr

    if step >= max_steps:
        return min_lr

    decay_steps = max(max_steps - warmup_steps, 1)
    decay_progress = (step - warmup_steps) / decay_steps
    decay_progress = min(max(decay_progress, 0.0), 1.0)
    cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr + cosine_coeff * (base_lr - min_lr)


def set_optimizer_lr(optimizer: AdamW, learning_rate: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def save_checkpoint(
    *,
    checkpoint_path: Path,
    model: DenseCausalLM,
    optimizer: AdamW,
    step: int,
    model_config: ModelConfig,
    data_config: dict[str, Any],
    train_config: dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "data_config": data_config,
            "train_config": train_config,
        },
        checkpoint_path,
    )


def append_loss_log(
    loss_log_path: Path,
    *,
    step: int,
    loss: float,
    tokens_per_sec: float,
    tokens_seen: int,
    learning_rate: float,
) -> None:
    loss_log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "step": step,
        "loss": loss,
        "tokens_per_sec": tokens_per_sec,
        "tokens_seen": tokens_seen,
        "learning_rate": learning_rate,
    }
    with loss_log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False))
        file.write("\n")


def load_checkpoint(
    checkpoint_path: Path,
    *,
    model: DenseCausalLM,
    optimizer: AdamW,
    device: torch.device,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])


def main() -> int:
    args = parse_args()

    try:
        model_config_data = load_yaml(args.model_config)
        data_config = load_yaml(args.data_config)
        train_config = load_yaml(args.train_config)

        if args.max_steps is not None:
            train_config["max_steps"] = args.max_steps
        if args.resume_from is not None:
            train_config["resume_from"] = str(args.resume_from)

        model_config = ModelConfig(**model_config_data)
        device = select_device(str(train_config.get("device", "auto")))
        set_seed(int(train_config.get("seed", 42)))

        dataset, dataloader = build_dataloader(data_config)
        check_config_compatibility(model_config, data_config, dataset)

        model = DenseCausalLM(model_config).to(device)
        optimizer = AdamW(
            model.parameters(),
            lr=float(train_config.get("learning_rate", 3e-4)),
            betas=(
                float(train_config.get("adam_beta1", 0.9)),
                float(train_config.get("adam_beta2", 0.95)),
            ),
            eps=float(train_config.get("adam_eps", 1e-8)),
            weight_decay=float(train_config.get("weight_decay", 0.1)),
        )

        start_step = 0
        resume_from = resolve_project_path(train_config.get("resume_from"))
        if resume_from is not None:
            start_step = load_checkpoint(
                resume_from,
                model=model,
                optimizer=optimizer,
                device=device,
            )

        output_dir = resolve_project_path(train_config.get("output_dir"))
        if output_dir is None:
            raise ValueError("train config must define output_dir")

        max_steps = int(train_config.get("max_steps", 1000))
        grad_accum_steps = int(train_config.get("gradient_accumulation_steps", 1))
        grad_clip = train_config.get("grad_clip", 1.0)
        log_interval = int(train_config.get("log_interval", 10))
        save_interval = int(train_config.get("save_interval", 500))
        save_latest = bool(train_config.get("save_latest", True))
        loss_log_file = train_config.get("loss_log_file", "losses.jsonl")
        loss_log_path = (
            output_dir / str(loss_log_file)
            if loss_log_file is not None
            else None
        )

        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if grad_accum_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if log_interval <= 0:
            raise ValueError("log_interval must be positive")
        get_learning_rate(start_step + 1, train_config)

        batch_iter = cycle_batches(dataloader)
        model.train()
        last_log_time = time.time()
        tokens_since_log = 0
        tokens_seen = (
            start_step
            * int(data_config.get("batch_size", 4))
            * dataset.block_size
            * grad_accum_steps
        )
        if start_step == 0 and loss_log_path is not None and loss_log_path.exists():
            loss_log_path.unlink()
        latest_path = output_dir / "latest.pt"
        if start_step == 0 and save_latest and latest_path.exists():
            latest_path.unlink()

        print(f"device: {device}")
        print(f"dataset_len: {len(dataset)}")
        print(f"batch_size: {data_config.get('batch_size', 4)}")
        print(f"block_size: {dataset.block_size}")
        print(f"max_steps: {max_steps}")
        print(f"start_step: {start_step}")
        print(f"learning_rate: {float(train_config.get('learning_rate', 3e-4))}")
        print(f"min_learning_rate: {float(train_config.get('min_learning_rate', 0.0))}")
        print(f"warmup_steps: {int(train_config.get('warmup_steps', 0))}")
        print(f"lr_decay: {bool(train_config.get('lr_decay', True))}")

        for step in range(start_step + 1, max_steps + 1):
            learning_rate = get_learning_rate(step, train_config)
            set_optimizer_lr(optimizer, learning_rate)
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            total_tokens = 0

            for _ in range(grad_accum_steps):
                batch = next(batch_iter)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                logits = model(input_ids)
                loss = causal_lm_loss(logits, labels)
                (loss / grad_accum_steps).backward()

                total_loss += float(loss.detach().cpu())
                total_tokens += int(input_ids.numel())

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            tokens_seen += total_tokens
            tokens_since_log += total_tokens

            if step == 1 or step % log_interval == 0:
                now = time.time()
                elapsed = max(now - last_log_time, 1e-9)
                tokens_per_sec = tokens_since_log / elapsed
                avg_loss = total_loss / grad_accum_steps
                print(
                    f"step={step:06d} loss={avg_loss:.4f} "
                    f"lr={learning_rate:.2e} tokens/s={tokens_per_sec:.0f}"
                )
                if loss_log_path is not None:
                    append_loss_log(
                        loss_log_path,
                        step=step,
                        loss=avg_loss,
                        tokens_per_sec=tokens_per_sec,
                        tokens_seen=tokens_seen,
                        learning_rate=float(learning_rate),
                    )
                last_log_time = now
                tokens_since_log = 0

            should_save_step = save_interval > 0 and step % save_interval == 0
            should_save_latest = save_latest and (should_save_step or step == max_steps)

            if should_save_step:
                checkpoint_path = output_dir / f"step_{step:06d}.pt"
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    model_config=model_config,
                    data_config=data_config,
                    train_config=train_config,
                )
                print(f"saved checkpoint: {checkpoint_path}")

            if should_save_latest:
                latest_path = output_dir / "latest.pt"
                save_checkpoint(
                    checkpoint_path=latest_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    model_config=model_config,
                    data_config=data_config,
                    train_config=train_config,
                )

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print("training finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
