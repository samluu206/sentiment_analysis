"""Model training utilities."""
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


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
        save_steps: int = 50,
        use_mlflow: bool = False,
        mlflow_tracker: Optional[any] = None
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
            use_mlflow: Whether to use MLflow tracking
            mlflow_tracker: MLflowTracker instance (required if use_mlflow=True)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        self.use_mlflow = use_mlflow
        self.mlflow_tracker = mlflow_tracker

        if use_mlflow and mlflow_tracker is None:
            raise ValueError("mlflow_tracker must be provided when use_mlflow=True")

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
    def compute_metrics(eval_pred, include_roc_auc: bool = True) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            eval_pred: Predictions and labels
            include_roc_auc: Whether to compute ROC-AUC score

        Returns:
            Dictionary of metrics
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average='binary'
        )
        accuracy = accuracy_score(labels, predictions)

        metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

        if include_roc_auc and logits.shape[1] == 2:
            probabilities = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
            try:
                roc_auc = roc_auc_score(labels, probabilities[:, 1])
                metrics['roc_auc'] = roc_auc
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")

        return metrics

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
        if self.use_mlflow:
            params = {
                "num_epochs": self.training_args.num_train_epochs,
                "batch_size": self.training_args.per_device_train_batch_size,
                "learning_rate": self.training_args.learning_rate,
                "weight_decay": self.training_args.weight_decay,
                "warmup_steps": self.training_args.warmup_steps,
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset),
                "model_name": self.model.config.name_or_path if hasattr(self.model.config, 'name_or_path') else "unknown"
            }
            self.mlflow_tracker.log_params(params)
            logger.info("Logged training parameters to MLflow")

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=self.data_collator
        )

        trainer.train()

        if self.use_mlflow:
            final_metrics = trainer.evaluate()
            self.mlflow_tracker.log_metrics({
                f"final_{k}": v for k, v in final_metrics.items()
                if isinstance(v, (int, float))
            })
            logger.info("Logged final metrics to MLflow")

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
