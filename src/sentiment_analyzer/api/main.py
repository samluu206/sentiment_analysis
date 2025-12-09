"""FastAPI application for sentiment analysis."""
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch

from sentiment_analyzer.inference.predictor import SentimentPredictor
from sentiment_analyzer.api.schemas import (
    PredictRequest,
    BatchPredictRequest,
    PredictResponse,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse
)


# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Production-ready sentiment analysis API using RoBERTa model (94.53% accuracy)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[SentimentPredictor] = None
MODEL_INFO = {
    "model_name": "roberta-base",
    "model_type": "RoBERTa",
    "accuracy": 0.9453,
    "f1_score": 0.9452,
    "roc_auc": 0.9828,
    "model_size_mb": 500,
    "num_labels": 2,
    "max_length": 128
}


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global predictor

    try:
        # Get model path from environment or use default
        model_path = os.getenv(
            "MODEL_PATH",
            str(Path(__file__).parent.parent.parent.parent / "models" / "roberta_sentiment")
        )

        print(f"Loading model from: {model_path}")

        # Auto-detect device (GPU if available, otherwise CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        if device == "cuda":
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPU detected, using CPU")

        # Load predictor
        predictor = SentimentPredictor(
            model_path=model_path,
            device=device
        )

        print("✅ Model loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs."""
    return {
        "message": "Sentiment Analysis API",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Check if the API and model are healthy"
)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        model_name=MODEL_INFO["model_name"]
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Get model information",
    description="Get details about the loaded model (architecture, performance metrics)"
)
async def get_model_info():
    """Get model information."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return ModelInfoResponse(**MODEL_INFO)


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Predict sentiment",
    description="Analyze sentiment of a single review text",
    responses={
        200: {"description": "Successful prediction"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Model not available"}
    }
)
async def predict(request: PredictRequest):
    """
    Predict sentiment for a single text.

    Args:
        request: PredictRequest with text field

    Returns:
        PredictResponse with sentiment, confidence, and probabilities

    Example:
        ```
        POST /predict
        {
            "text": "This product is amazing!"
        }

        Response:
        {
            "sentiment": "POSITIVE",
            "confidence": 0.96,
            "probabilities": {
                "NEGATIVE": 0.04,
                "POSITIVE": 0.96
            }
        }
        ```
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        # Make prediction
        result = predictor.predict_with_confidence(request.text)

        # Format response
        return PredictResponse(
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            probabilities={
                "NEGATIVE": result["negative_prob"],
                "POSITIVE": result["positive_prob"]
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/batch_predict",
    response_model=BatchPredictResponse,
    tags=["Prediction"],
    summary="Batch predict sentiments",
    description="Analyze sentiment of multiple review texts (up to 100)",
    responses={
        200: {"description": "Successful predictions"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Model not available"}
    }
)
async def batch_predict(request: BatchPredictRequest):
    """
    Predict sentiment for multiple texts.

    Args:
        request: BatchPredictRequest with texts field

    Returns:
        BatchPredictResponse with list of predictions

    Example:
        ```
        POST /batch_predict
        {
            "texts": [
                "Great product!",
                "Terrible quality."
            ]
        }

        Response:
        {
            "predictions": [
                {
                    "sentiment": "POSITIVE",
                    "confidence": 0.95,
                    "probabilities": {"NEGATIVE": 0.05, "POSITIVE": 0.95}
                },
                {
                    "sentiment": "NEGATIVE",
                    "confidence": 0.88,
                    "probabilities": {"NEGATIVE": 0.88, "POSITIVE": 0.12}
                }
            ],
            "count": 2
        }
        ```
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        # Make batch prediction
        results = predictor.batch_predict(request.texts, batch_size=32)

        # Format predictions
        predictions = [
            PredictResponse(
                sentiment=result["sentiment"],
                confidence=result["confidence"],
                probabilities={
                    "NEGATIVE": result["negative_prob"],
                    "POSITIVE": result["positive_prob"]
                }
            )
            for result in results
        ]

        return BatchPredictResponse(
            predictions=predictions,
            count=len(predictions)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
