"""Model initialization and management."""
from transformers import AutoModelForSequenceClassification
import torch
from pathlib import Path
from typing import Optional

try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False


class SentimentModel:
    """Manages sentiment classification model."""

    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        device: Optional[str] = None
    ):
        """
        Initialize model.

        Args:
            model_name: Name of the pretrained model or path to saved model
            num_labels: Number of classification labels
            device: Device to load model on (cpu/cuda/dml). Auto-detected if None.
                   Priority: DirectML (AMD GPU) > CUDA (NVIDIA GPU) > CPU
        """
        self.model_name = model_name
        self.num_labels = num_labels

        if device:
            self.device = device
        elif HAS_DIRECTML:
            self.device = torch_directml.device()
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = None

    def load_model(self):
        """Load or initialize the model."""
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                ignore_mismatched_sizes=True,
                use_safetensors=True
            )
        except (OSError, ValueError):
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                ignore_mismatched_sizes=True
            )
        self.model.to(self.device)
        return self.model

    def save_model(self, save_path: str):
        """
        Save model to disk.

        Args:
            save_path: Directory path to save model
        """
        if self.model is None:
            raise ValueError("No model loaded to save")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path, safe_serialization=True)

    def get_model(self):
        """Return the model instance."""
        if self.model is None:
            self.load_model()
        return self.model

    def get_num_parameters(self) -> int:
        """Get total number of model parameters."""
        if self.model is None:
            self.load_model()
        return sum(p.numel() for p in self.model.parameters())

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        if self.model is None:
            self.load_model()
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
