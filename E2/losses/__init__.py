from losses.infonce import InfoNCELoss

# SINCERE is deprecated and lives at losses/sincere_deprecated.py for
# historical reference only. The 2026-05 restructure moved every CL method
# (CL-FT, CLIP-CL, CL+CE) to InfoNCE for consistency. Do not import SINCERE
# in new code.

__all__ = ["InfoNCELoss"]
