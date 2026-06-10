"""Contrastive normalization: subtract global embedding, then L2-normalize.

The global (song-level) embedding captures what "this song sounds like overall"
— a direction that dominates all window embeddings and makes unrelated sections
appear highly similar.  Subtracting it before L2-normalizing exposes the
residual per-section variation that drives alignment.
"""

from __future__ import annotations

import torch


def fit_global(raw_win_embs: torch.Tensor) -> torch.Tensor:
    """
    Compute the song-level mean embedding from all dense window embeddings.

    Args:
        raw_win_embs: [N, D] un-normalized window means

    Returns:
        [D] global embedding vector
    """
    return raw_win_embs.mean(dim=0)


def apply_contrastive(
    embs: torch.Tensor,       # [N, D] or [D]
    global_emb: torch.Tensor, # [D]
) -> torch.Tensor:
    """
    Subtract the global embedding and L2-normalize each row.

    Works for both batched [N, D] and single [D] inputs.

    Returns:
        Same shape as input, L2-normalized after subtraction.
    """
    single = embs.dim() == 1
    if single:
        embs = embs.unsqueeze(0)

    X = embs - global_emb.unsqueeze(0)
    X = X / (X.norm(dim=1, keepdim=True) + 1e-9)

    return X.squeeze(0) if single else X
