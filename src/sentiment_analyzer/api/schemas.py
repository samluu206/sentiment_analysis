"""Pydantic schemas for API request/response validation."""
from pydantic import BaseModel, Field, validator
from typing import List, Literal


class PredictRequest(BaseModel):
    """Request schema for single prediction."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Review text to analyze (1-5000 characters)",
        example="This product exceeded my expectations! Highly recommend."
    )

    @validator('text')
    def text_not_empty(cls, v):
        """Validate text is not just whitespace."""
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()


class BatchPredictRequest(BaseModel):
    """Request schema for batch prediction."""

    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of review texts (1-100 items)",
        example=[
            "Great product!",
            "Terrible quality.",
            "Average, nothing special."
        ]
    )

    @validator('texts')
    def validate_texts(cls, v):
        """Validate all texts are non-empty."""
        cleaned = []
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} is empty")
            if len(text) > 5000:
                raise ValueError(f"Text at index {i} exceeds 5000 characters")
            cleaned.append(text.strip())
        return cleaned


class PredictResponse(BaseModel):
    """Response schema for single prediction."""

    sentiment: Literal["POSITIVE", "NEGATIVE"] = Field(
        ...,
        description="Predicted sentiment"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1)"
    )
    probabilities: dict = Field(
        ...,
        description="Probability for each class",
        example={"NEGATIVE": 0.12, "POSITIVE": 0.88}
    )

    class Config:
        schema_extra = {
            "example": {
                "sentiment": "POSITIVE",
                "confidence": 0.88,
                "probabilities": {
                    "NEGATIVE": 0.12,
                    "POSITIVE": 0.88
                }
            }
        }


class BatchPredictResponse(BaseModel):
    """Response schema for batch prediction."""

    predictions: List[PredictResponse] = Field(
        ...,
        description="List of predictions"
    )
    count: int = Field(
        ...,
        description="Number of predictions"
    )

    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "sentiment": "POSITIVE",
                        "confidence": 0.95,
                        "probabilities": {"NEGATIVE": 0.05, "POSITIVE": 0.95}
                    },
                    {
                        "sentiment": "NEGATIVE",
                        "confidence": 0.82,
                        "probabilities": {"NEGATIVE": 0.82, "POSITIVE": 0.18}
                    }
                ],
                "count": 2
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: Literal["healthy", "unhealthy"] = Field(
        ...,
        description="Service health status"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether model is loaded"
    )
    model_name: str = Field(
        ...,
        description="Name of loaded model"
    )


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""

    model_name: str = Field(..., description="Model identifier")
    model_type: str = Field(..., description="Model architecture")
    accuracy: float = Field(..., description="Test accuracy")
    f1_score: float = Field(..., description="F1 score")
    roc_auc: float = Field(..., description="ROC-AUC score")
    model_size_mb: int = Field(..., description="Model size in MB")
    num_labels: int = Field(..., description="Number of output classes")
    max_length: int = Field(..., description="Maximum sequence length")

    class Config:
        schema_extra = {
            "example": {
                "model_name": "roberta-base",
                "model_type": "RoBERTa",
                "accuracy": 0.9453,
                "f1_score": 0.9452,
                "roc_auc": 0.9828,
                "model_size_mb": 500,
                "num_labels": 2,
                "max_length": 128
            }
        }


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: str = Field(None, description="Additional error details")

    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Text cannot be empty",
                "detail": "Please provide valid review text"
            }
        }
