"""Configuration scaffold for the LS ML Backend service.

Holds placeholders for settings such as model paths, thresholds, port,
batch size, etc. No functionality is implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    """Placeholder settings container."""

    model_config_path: Optional[str] = None
    model_checkpoint_path: Optional[str] = None
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    server_port: int = 9090
    batch_size: int = 1


# Default settings instance (may be overridden in actual implementation)
DEFAULT_SETTINGS = Settings()
