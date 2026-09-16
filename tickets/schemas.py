"""Request/response contracts. Text needs stricter edge validation than numbers do."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from tickets.config import MAX_TEXT_CHARS, MIN_TEXT_CHARS


class PredictionRequest(BaseModel):
    model_config = {"extra": "forbid"}

    text: str = Field(
        min_length=MIN_TEXT_CHARS,
        max_length=MAX_TEXT_CHARS,
        examples=["hi team, i was charged twice for my subscription this month"],
    )

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        """A max_length cap bounds memory and cost; a blank check stops the model
        being asked to classify nothing at all, which it will happily do."""
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    label: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_version: str
    model_loaded: bool
