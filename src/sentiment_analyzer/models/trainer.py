"""Model training utilities."""
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict


class SentimentTrainer:
    """Handles model training and evaluation."""

    def __init__(
        self,
        model,
        tokenizer,
        output_dir: str = "results",
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        logging_steps: int = 10,
        eval_steps: int = 50,
        save_steps: int = 50
    ):
        """
        Initialize trainer.

        Args:
            model: The model to train
            tokenizer: Tokenizer for data collation
            output_dir: Directory for outputs
            num_epochs: Number of training epochs
            batch_size: Batch size for training and evaluation
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps
            logging_steps: Log every N steps
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
        """
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=3,
            report_to="none"
        )

    @staticmethod
    def compute_metrics(eval_pred) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            eval_pred: Predictions and labels

        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average='binary'
        )
        accuracy = accuracy_score(labels, predictions)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset
    ):
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset

        Returns:
            Trainer instance
        """
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=self.data_collator
        )

        trainer.train()
        return trainer

    def evaluate(self, trainer, eval_dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            trainer: Trained Trainer instance
            eval_dataset: Dataset to evaluate on

        Returns:
            Dictionary of evaluation metrics
        """
        return trainer.evaluate(eval_dataset)
