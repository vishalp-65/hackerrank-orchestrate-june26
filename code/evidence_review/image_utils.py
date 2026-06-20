"""Image normalization for the vision API.

The dataset ships JPEG, PNG, WEBP, and **AVIF** files all carrying a ``.jpg``
extension. AVIF is not accepted by the Anthropic vision API, so every image is
decoded with Pillow (AVIF/HEIC plugins registered when available), downscaled, and
re-encoded to JPEG. This yields one uniform ``image/jpeg`` path and caps image-token
cost (Anthropic charges ~ width*height/750 tokens per image).
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

from . import config

# Register decoders for the container formats the dataset hides behind .jpg.
try:  # AVIF (8 test images) — required
    import pillow_avif  # noqa: F401  (import has the side effect of registering)
    _AVIF_OK = True
except Exception:  # pragma: no cover
    _AVIF_OK = False

try:  # HEIC — defensive (none in this dataset, but cheap insurance)
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pass


def sniff_format(path: Path) -> str:
    """Return the true container format from magic bytes (ignores the extension)."""
    try:
        with open(path, "rb") as f:
            h = f.read(32)
    except OSError:
        return "MISSING"
    if h[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if h[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if h[:4] == b"RIFF" and h[8:12] == b"WEBP":
        return "WEBP"
    if h[:6] in (b"GIF87a", b"GIF89a"):
        return "GIF"
    if h[4:8] == b"ftyp":
        brand = h[8:12]
        if brand == b"avif":
            return "AVIF"
        if brand in (b"heic", b"heix", b"mif1", b"hevc"):
            return "HEIC"
        return "ISOBMFF"
    return "UNKNOWN"


def normalize_to_jpeg_b64(path: Path, max_edge: int = config.MAX_IMAGE_EDGE,
                          quality: int = config.JPEG_QUALITY) -> dict | None:
    """Decode any supported format → downscale → JPEG base64.

    Returns ``{"data", "media_type", "width", "height", "src_format", "est_tokens"}``
    or ``None`` if the image cannot be decoded.
    """
    src_format = sniff_format(path)
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            longest = max(w, h)
            if longest > max_edge:
                scale = max_edge / float(longest)
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                               Image.Resampling.LANCZOS)
            out_w, out_h = im.size
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
    except Exception:
        return None
    return {
        "data": base64.standard_b64encode(buf.getvalue()).decode("ascii"),
        "media_type": "image/jpeg",
        "width": out_w,
        "height": out_h,
        "src_format": src_format,
        "est_tokens": (out_w * out_h) // 750,
    }
