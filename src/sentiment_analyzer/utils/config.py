"""Configuration management for sentiment analyzer."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    """Data configuration."""
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    sample_size: Optional[int] = None
    train_split: float = 0.6
    val_split: float = 0.2
    test_split: float = 0.2
    random_seed: int = 42

    def __post_init__(self):
        """Validate that splits sum to 1.0."""
        total = self.train_split + self.val_split + self.test_split
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(
                f"Data splits must sum to 1.0, got {total:.3f} "
                f"(train={self.train_split}, val={self.val_split}, test={self.test_split})"
            )


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    num_labels: int = 2
    max_length: int = 128
    model_save_path: str = "models/trained"


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    output_dir: str = "results"


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()

    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent.parent.parent
