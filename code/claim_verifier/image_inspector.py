"""Local, non-authoritative image diagnostics and preprocessing."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from claim_verifier.models import ImageDiagnostics


@dataclass(frozen=True)
class PreparedImage:
    image_id: str
    path: str
    data_url: str
    diagnostics: ImageDiagnostics


class ImageInspector:
    def __init__(self, max_long_edge: int = 1600, jpeg_quality: int = 85) -> None:
        self.max_long_edge = max_long_edge
        self.jpeg_quality = jpeg_quality

    def prepare(self, image_path: str | Path) -> PreparedImage:
        path = Path(image_path)
        image_id = path.stem
        try:
            with Image.open(path) as image:
                original = image.convert("RGB")
                fmt = image.format or path.suffix.lstrip(".").upper() or "UNKNOWN"
        except Exception as exc:
            raise ValueError(f"Cannot open image {path}: {exc}") from exc

        preprocessed = _resize_preserving_aspect(original, self.max_long_edge)
        buffer = io.BytesIO()
        preprocessed.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
        payload = buffer.getvalue()

        gray = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_luminance = float(np.mean(gray))
        contrast = float(np.std(gray))

        diagnostics = ImageDiagnostics(
            image_id=image_id,
            path=str(path),
            file_size_bytes=path.stat().st_size,
            width=original.width,
            height=original.height,
            format=fmt,
            mean_luminance=mean_luminance,
            contrast=contrast,
            blur_score=blur_score,
            perceptual_hash=_average_hash(original),
            preprocessed_width=preprocessed.width,
            preprocessed_height=preprocessed.height,
            preprocessed_size_bytes=len(payload),
        )
        data_url = "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")
        return PreparedImage(image_id=image_id, path=str(path), data_url=data_url, diagnostics=diagnostics)


def _resize_preserving_aspect(image: Image.Image, max_long_edge: int) -> Image.Image:
    long_edge = max(image.width, image.height)
    if long_edge <= max_long_edge:
        return image.copy()
    scale = max_long_edge / long_edge
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _average_hash(image: Image.Image, size: int = 8) -> str:
    small = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.array(small)
    average = pixels.mean()
    bits = "".join("1" if pixel > average else "0" for pixel in pixels.flatten())
    return f"{int(bits, 2):0{size * size // 4}x}"
