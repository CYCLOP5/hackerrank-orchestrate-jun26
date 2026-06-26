"""Fail-loud startup validation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PIL import Image

from claim_verifier.io import read_claims


def preflight(
    *,
    claims_path: str | Path,
    history_path: str | Path,
    requirements_path: str | Path,
    output_path: str | Path,
    require_openai: bool,
    limit: int | None = None,
) -> None:
    if require_openai:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required")
        if not os.environ.get("OPENAI_VISION_MODEL", "gpt-5.5").strip():
            raise RuntimeError("OPENAI_VISION_MODEL must not be empty")
        if not os.environ.get("OPENAI_AUDITOR_MODEL", "gpt-5.4-mini").strip():
            raise RuntimeError("OPENAI_AUDITOR_MODEL must not be empty")

    for path in (claims_path, history_path, requirements_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Required file does not exist: {path}")

    claims = read_claims(claims_path, limit=limit)
    for claim in claims:
        for image_path in claim.image_paths:
            path = Path(image_path)
            if not path.is_file():
                raise FileNotFoundError(f"Missing image for row {claim.row_index}: {image_path}")
            _verify_image(path, claim.row_index)

    _check_output_writable(output_path)


def _verify_image(path: Path, row_index: int) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(f"Corrupt/unreadable image for row {row_index}: {path}: {exc}") from exc


def _check_output_writable(output_path: str | Path) -> None:
    destination = Path(output_path)
    directory = destination.parent if str(destination.parent) else Path(".")
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, delete=True) as handle:
        handle.write(b"ok")
