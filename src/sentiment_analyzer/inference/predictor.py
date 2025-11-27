"""Inference and prediction utilities."""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Union, List, Dict
from pathlib import Path


class SentimentPredictor:
    """Handles sentiment prediction on new text."""

    def __init__(
        self,
        model_path: str,
        device: str = None
    ):
        """
        Initialize predictor with trained model.

        Args:
            model_path: Path to saved model directory
            device: Device to run inference on (cpu/cuda). Auto-detected if None.
        """
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path)
        )
        self.model.to(self.device)
        self.model.eval()

        self.label_map = {0: "NEGATIVE", 1: "POSITIVE"}

    def predict(
        self,
        text: Union[str, List[str]],
        return_probabilities: bool = False
    ) -> Union[str, List[str], Dict]:
        """
        Predict sentiment for input text.

        Args:
            text: Single text or list of texts
            return_probabilities: If True, return probabilities alongside predictions

        Returns:
            Predicted sentiment(s) or dict with predictions and probabilities
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

        predicted_labels = [self.label_map[pred.item()] for pred in predictions]

        if is_single:
            predicted_labels = predicted_labels[0]

        if return_probabilities:
            probs = probabilities.cpu().numpy()
            return {
                "predictions": predicted_labels,
                "probabilities": probs.tolist() if not is_single else probs[0].tolist(),
                "confidence": probs.max(axis=-1).tolist() if not is_single else float(probs[0].max())
            }

        return predicted_labels

    def predict_with_confidence(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Predict sentiment with confidence score.

        Args:
            text: Input text

        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1)
            confidence = probabilities.max().item()

        probs = probabilities.cpu().numpy()[0]

        return {
            "sentiment": self.label_map[prediction.item()],
            "confidence": confidence,
            "negative_prob": float(probs[0]),
            "positive_prob": float(probs[1])
        }

    def batch_predict(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Predict sentiment for a batch of texts efficiently.

        Args:
            texts: List of texts to predict
            batch_size: Batch size for processing

        Returns:
            List of prediction dictionaries
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.predict(batch, return_probabilities=True)

            for j, pred in enumerate(batch_results["predictions"]):
                results.append({
                    "sentiment": pred,
                    "confidence": batch_results["confidence"][j],
                    "negative_prob": batch_results["probabilities"][j][0],
                    "positive_prob": batch_results["probabilities"][j][1]
                })

        return results
