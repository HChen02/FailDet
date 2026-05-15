"""InfoNCE loss — single class with two modes.

Used by all three contrastive methods in the project:

- CL-FT  : supervised mode (image embeddings, class labels). Same-class
           samples are positives; different-class are negatives. Equivalent
           to standard supervised contrastive (SupCon) without the
           same-class-in-denominator subtlety SINCERE addressed; we keep
           it simple and uniform across methods.
- CL+CE  : supervised mode on [EOS] hidden-state embeddings.
- CLIP-CL: CLIP mode (image and text embeddings). Each image-text pair
           shares an index, all other off-diagonal pairs are negatives.
           Symmetric (image→text + text→image, averaged).

Both modes are vectorized — no per-anchor loops at runtime.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Unified InfoNCE used by CL-FT, CL+CE, and CLIP-CL.

    Args:
        temperature: scalar τ. Forward arguments can override this with
            a learnable temperature (CLIP-style) by passing ``temp_override``.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        temp_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Supervised mode (CL-FT, CL+CE) — call with features_a + labels:
            features_a: [B, D] L2-normalized embeddings
            labels:     [B]   class indices

        CLIP mode — call with features_a + features_b:
            features_a: [B, D] image embeddings (L2-normalized)
            features_b: [B, D] text embeddings (L2-normalized)

        ``temp_override`` (optional) is a 0-d tensor; useful for the
        learnable temperature in CLIP-CL (e.g. ``logit_scale.exp()``).
        """
        if not torch.isfinite(features_a).all():
            raise ValueError("InfoNCELoss received non-finite features_a")
        # Use float32 for numerical stability in the similarity matrix.
        features_a = features_a.float()
        tau = (temp_override.float()
               if temp_override is not None
               else torch.tensor(self.temperature, device=features_a.device,
                                 dtype=features_a.dtype))

        if features_b is not None:
            return self._forward_clip(features_a, features_b.float(), tau)

        if labels is None:
            raise ValueError(
                "InfoNCELoss supervised mode requires `labels`; pass "
                "`features_b` for CLIP mode instead."
            )
        return self._forward_supervised(features_a, labels, tau)

    @staticmethod
    def _forward_clip(
        feats_a: torch.Tensor, feats_b: torch.Tensor, tau: torch.Tensor,
    ) -> torch.Tensor:
        # Symmetric InfoNCE on the diagonal alignment.
        logits = feats_a @ feats_b.T / tau  # [B,B]
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.T, targets)
        return 0.5 * (loss_i2t + loss_t2i)

    @staticmethod
    def _forward_supervised(
        features: torch.Tensor, labels: torch.Tensor, tau: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized supervised InfoNCE.

        For each anchor i with positive set P(i)={j!=i : y_j==y_i}:
            loss_i = (1/|P(i)|) * Σ_{p∈P(i)} [ -sim(i,p) + logsumexp_{j!=i} sim(i,j) ]
        Returns mean over anchors that have at least one positive.

        Equivalently (vectorized):
            per_anchor = |P(i)| * log_denom[i] - Σ_{p∈P(i)} sim[i,p]
            total      = Σ_i per_anchor[i]
            loss       = total / Σ_i |P(i)|
        """
        device = features.device
        B = features.size(0)
        sim = features @ features.T / tau  # [B,B]

        self_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B,B]
        pos_mask = labels_eq & self_mask                        # [B,B]

        # Denominator: log Σ_{j!=i} exp(sim[i,j])  — includes positives too,
        # matching standard supervised contrastive (SupCon-style).
        sim_for_denom = sim.masked_fill(~self_mask, float("-inf"))
        log_denom = torch.logsumexp(sim_for_denom, dim=1)  # [B]

        # Numerator: Σ_{p∈P(i)} sim[i,p]
        sim_pos = sim.masked_fill(~pos_mask, 0.0)
        pos_sum = sim_pos.sum(dim=1)                        # [B]
        pos_count = pos_mask.sum(dim=1)                     # [B] (long)

        per_anchor = pos_count.float() * log_denom - pos_sum  # [B]
        total_pairs = pos_count.sum()
        if total_pairs.item() == 0:
            # Degenerate batch (no same-class pairs anywhere). The CL+CE
            # runner asserts batch_size>=4 to avoid this; for CL-FT we
            # return 0 with grad to keep training alive.
            return torch.zeros((), device=device, dtype=features.dtype,
                               requires_grad=True)
        return per_anchor.sum() / total_pairs.float()


__all__ = ["InfoNCELoss"]
