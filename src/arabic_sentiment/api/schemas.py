"""
Pydantic schemas for API request/response validation.

Provides automatic validation, documentation, and type safety
for all FastAPI endpoints.
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """What the user sends to /predict."""

    text: str = Field(
        ...,
        min_length=1,
        description="Arabic text to classify for sentiment",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"text": "المنتج رائع والتوصيل سريع"}]
        }
    }


class PredictionResponse(BaseModel):
    """What /predict returns."""

    label: str
    confidence: float
    model_version: str
    correlation_id: str
    latency_ms: float


class HealthResponse(BaseModel):
    """What /health returns."""

    status: str
    model_loaded: bool
    model_version: str


class MetadataResponse(BaseModel):
    """What /metadata returns."""

    model_version: str
    model_name: str
    framework: str
    max_seq_length: int
