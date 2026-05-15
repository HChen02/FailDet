"""Image corruptions for E7 robustness experiment.

5 corruption types × 5 severity levels = 25 variants.

Severity values approximate the ImageNet-C / Hendrycks 2019 levels but
were chosen for Guardian-Fail's 256×256 images and the ImageNet
normalisation the encoders use.

Each corruption is deterministic given (severity, image-content): seeded
per call so the same test image always gets the same corruption.
"""

from __future__ import annotations

import io
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def gaussian_noise(img: Image.Image, severity: int, *, rng_seed: int = 0) -> Image.Image:
    """Pixel-additive Gaussian noise."""
    stds = [0.04, 0.06, 0.08, 0.09, 0.10]
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    rng = np.random.default_rng(rng_seed)
    arr = arr + rng.normal(0.0, stds[severity - 1], arr.shape)
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))


def gaussian_blur(img: Image.Image, severity: int, **_) -> Image.Image:
    sigmas = [1.0, 2.0, 3.0, 4.0, 6.0]
    return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=sigmas[severity - 1]))


def jpeg_compression(img: Image.Image, severity: int, **_) -> Image.Image:
    qualities = [25, 18, 15, 10, 7]
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=qualities[severity - 1])
    buf.seek(0)
    return Image.open(buf).copy()


def contrast(img: Image.Image, severity: int, **_) -> Image.Image:
    """Reduce contrast (lower factor = greyer image)."""
    factors = [0.75, 0.50, 0.40, 0.30, 0.15]
    return ImageEnhance.Contrast(img.convert("RGB")).enhance(factors[severity - 1])


def brightness(img: Image.Image, severity: int, **_) -> Image.Image:
    """Reduce brightness (darken). Severity 5 is near-black."""
    factors = [0.70, 0.60, 0.50, 0.40, 0.30]
    return ImageEnhance.Brightness(img.convert("RGB")).enhance(factors[severity - 1])


CORRUPTIONS: dict[str, Callable[[Image.Image, int], Image.Image]] = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur":  gaussian_blur,
    "jpeg":           jpeg_compression,
    "contrast":       contrast,
    "brightness":     brightness,
}


def apply(img: Image.Image,
          name: Optional[str],
          severity: Optional[int],
          *, rng_seed: int = 0) -> Image.Image:
    """Public entry — no-op when name or severity is None."""
    if not name or not severity:
        return img
    if name not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption: {name!r}. "
                         f"Known: {list(CORRUPTIONS.keys())}")
    if not (1 <= int(severity) <= 5):
        raise ValueError(f"severity must be in [1, 5], got {severity}")
    return CORRUPTIONS[name](img, int(severity), rng_seed=rng_seed)


# Pretty-print labels for tables.
CORRUPTION_DISPLAY = {
    "gaussian_noise": "Gauss-Noise",
    "gaussian_blur":  "Gauss-Blur",
    "jpeg":           "JPEG",
    "contrast":       "Contrast",
    "brightness":     "Brightness",
}
