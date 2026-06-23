import torch
import torch.nn.functional as F

def causal_lm_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int=-100,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(
            f"logits应为[B, T, V], 实际 shape={logits.shape}"
        )
    if labels.ndim != 2:
        raise ValueError(
            f"labels应为[B, T], 实际 shape={labels.shape}"
        )
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "logits的前两维必须与labels一致，"
            f"实际 logits={logits.shape}, labels={labels.shape}"
        )
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    vocab_size = shift_logits.shape[-1]

    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    loss = F.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=ignore_index
    )
    return loss