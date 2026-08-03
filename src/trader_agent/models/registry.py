"""Model registry -- versioned storage and loading of trained model artifacts."""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_MODEL_DIR = Path("artifacts/models")


def save_model(model, version: str, path: Path = DEFAULT_MODEL_DIR) -> Path:
    """Serialize and save a trained model with version metadata."""
    raise NotImplementedError


def load_model(version: str = "latest", path: Path = DEFAULT_MODEL_DIR):
    """Load a model artifact by version tag."""
    raise NotImplementedError
