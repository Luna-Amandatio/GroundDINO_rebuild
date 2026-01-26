"""Pydantic schema scaffolding for request/response models.

No functionality is implemented yet. Actual Pydantic models will be defined
in a future task. Using simple placeholders to avoid introducing dependencies
at this stage.
"""

from __future__ import annotations

from typing import Any, Dict


class PredictionRequest:  # Placeholder; will be replaced with Pydantic BaseModel
    """Represents an incoming prediction request (placeholder)."""

    def __init__(self, data: Dict[str, Any] | None = None) -> None:
        self.data = data or {}


class PredictionResponse:  # Placeholder; will be replaced with Pydantic BaseModel
    """Represents a prediction response (placeholder)."""

    def __init__(self, result: Dict[str, Any] | None = None) -> None:
        self.result = result or {}
