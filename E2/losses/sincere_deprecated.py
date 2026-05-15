"""
SINCERE — Supervised InfoNCE without intra-class repulsion.

Reference:
  Feeney & Hughes (2024), "SINCERE: Supervised Information Noise-Contrastive
  Estimation REvisited" (https://arxiv.org/abs/2309.14277).

Difference from SupCon (Khosla et al. 2020):
  SupCon's denominator includes both same-class and different-class pairs,
  so same-class anchors compete with each other for "uniqueness." SINCERE's
  denominator is built from negatives only, so positives are pulled together
  without an opposing intra-class repulsion term. This is closer to the
  classical InfoNCE form and tends to be more stable when the encoder starts
  with collapsed/low-variance features (the "cold-start" regime).

Per-anchor loss (anchor i, positives P_i, negatives N_i, sims s_ij = z_i·z_j/τ):

    L_i = -log( sum_{p in P_i} exp(s_ip)  /  ( sum_{p in P_i} exp(s_ip)
                                              + sum_{n in N_i} exp(s_in) ) )

Anchors with no positives (their class appears once in the batch) are
skipped, mirroring SupCon's standard practice.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SINCERELoss(nn.Module):
    """SINCERE loss: supervised InfoNCE with negatives-only denominator.

    The forward computes everything in float32 even if `features` arrive as
    bf16/fp16 — the contrastive softmax has small per-pair gradients that
    underflow in low-precision mantissas and stall training.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: [B, D] — assumed L2-normalized (caller handles normalize).
        # labels:   [B]  — integer class ids.
        if not torch.isfinite(features).all():
            raise ValueError(
                "SINCERELoss received non-finite features (NaN or Inf). "
                "An upstream encoder produced bad activations — fix the "
                "encoder before continuing."
            )
        features = features.float()
        device = features.device
        b = features.shape[0]

        sim = features @ features.T / self.temperature

        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)        # [B, B]
        self_mask = ~torch.eye(b, dtype=torch.bool, device=device)    # exclude diag

        pos_mask = labels_eq & self_mask                              # same class, not self
        neg_mask = ~labels_eq                                         # different class

        # Mask out non-pos / non-neg entries with -inf so they vanish under logsumexp.
        pos_sim = sim.masked_fill(~pos_mask, float("-inf"))
        neg_sim = sim.masked_fill(~neg_mask, float("-inf"))

        log_sum_pos = torch.logsumexp(pos_sim, dim=1)                 # [B]
        log_sum_neg = torch.logsumexp(neg_sim, dim=1)                 # [B]

        # log( exp(log_sum_pos) + exp(log_sum_neg) ) — stable form of denom.
        log_denom = torch.logsumexp(
            torch.stack([log_sum_pos, log_sum_neg], dim=1), dim=1
        )

        # Skip anchors with no positives (class singleton in batch).
        has_pos = pos_mask.any(dim=1)
        if not has_pos.any():
            return sim.new_zeros(())

        per_anchor = -(log_sum_pos - log_denom)
        return per_anchor[has_pos].mean()
