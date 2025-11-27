"""Helper utilities for the sentiment analyzer."""
import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


def setup_logging(log_file: str = None, level: int = logging.INFO):
    """
    Set up logging configuration.

    Args:
        log_file: Optional file path to save logs
        level: Logging level
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def save_metrics(metrics: Dict[str, Any], output_path: str):
    """
    Save metrics to JSON file.

    Args:
        metrics: Dictionary of metrics
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(metrics_path: str) -> Dict[str, Any]:
    """
    Load metrics from JSON file.

    Args:
        metrics_path: Path to metrics JSON file

    Returns:
        Dictionary of metrics
    """
    with open(metrics_path, 'r') as f:
        return json.load(f)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list = None,
    save_path: str = None
):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        save_path: Optional path to save figure
    """
    if labels is None:
        labels = ["Negative", "Positive"]

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    plt.close()


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list = None,
    output_dict: bool = False
):
    """
    Generate classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        output_dict: If True, return as dictionary

    Returns:
        Classification report as string or dict
    """
    if labels is None:
        labels = ["Negative", "Positive"]

    return classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=output_dict
    )


def analyze_predictions(
    df: pd.DataFrame,
    text_column: str = "text",
    true_column: str = "true_label",
    pred_column: str = "predicted_label",
    n_samples: int = 10
) -> pd.DataFrame:
    """
    Analyze misclassified predictions.

    Args:
        df: DataFrame with texts, true labels, and predictions
        text_column: Name of text column
        true_column: Name of true label column
        pred_column: Name of predicted label column
        n_samples: Number of misclassified samples to return

    Returns:
        DataFrame with misclassified examples
    """
    misclassified = df[df[true_column] != df[pred_column]]

    if len(misclassified) == 0:
        return pd.DataFrame()

    return misclassified[[text_column, true_column, pred_column]].head(n_samples)


def print_model_info(model):
    """
    Print model architecture information.

    Args:
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
