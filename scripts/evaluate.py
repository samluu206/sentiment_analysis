#!/usr/bin/env python3
"""
Model Evaluation Script

Comprehensive evaluation of trained sentiment analysis model on held-out test set.
Generates confusion matrix, classification report, ROC curves, and detailed metrics.

Usage:
    python scripts/evaluate.py --model-path models/trained_model_v2
    python scripts/evaluate.py --model-path models/trained_model_v2 --output-dir evaluation_results
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from datasets import Dataset

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sentiment_analyzer.data.data_loader import SentimentDataLoader
from src.sentiment_analyzer.data.preprocessor import TextPreprocessor
from src.sentiment_analyzer.inference.predictor import SentimentPredictor


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate sentiment analysis model")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/amazon_polarity_20k.csv",
        help="Path to dataset CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def load_test_data(data_path: str, random_seed: int = 42) -> Dataset:
    """Load held-out test set"""
    print(f"Loading test data from {data_path}...")

    loader = SentimentDataLoader(
        data_path=data_path,
        train_split=0.6,
        test_split=0.2,
        random_seed=random_seed,
    )

    _, _, test_dataset = loader.load_and_prepare()

    print(f"Test set size: {len(test_dataset)} samples")
    return test_dataset


def predict_on_dataset(
    predictor: SentimentPredictor, dataset: Dataset
) -> tuple[list, list, list]:
    """
    Run predictions on entire dataset

    Returns:
        tuple: (true_labels, predicted_labels, probabilities)
    """
    print("\nRunning predictions on test set...")

    true_labels = []
    predicted_labels = []
    all_probabilities = []

    for i, example in enumerate(dataset):
        if i % 500 == 0:
            print(f"  Processed {i}/{len(dataset)} samples...")

        text = example["text"]
        true_label = example["label"]

        result = predictor.predict_with_confidence(text)
        pred_label = 1 if result["sentiment"] == "POSITIVE" else 0

        true_labels.append(true_label)
        predicted_labels.append(pred_label)
        all_probabilities.append(result["probabilities"])

    print(f"  Processed {len(dataset)}/{len(dataset)} samples. Done!")

    return true_labels, predicted_labels, all_probabilities


def calculate_metrics(
    true_labels: list, predicted_labels: list, probabilities: list
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics"""
    print("\nCalculating metrics...")

    # Convert to numpy arrays
    y_true = np.array(true_labels)
    y_pred = np.array(predicted_labels)
    y_proba = np.array(probabilities)

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = (
        precision_recall_fscore_support(y_true, y_pred, average=None)
    )

    # ROC-AUC (using probability of positive class)
    try:
        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
    except Exception as e:
        print(f"Warning: Could not calculate ROC-AUC: {e}")
        roc_auc = None

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "per_class": {
            "negative": {
                "precision": float(precision_per_class[0]),
                "recall": float(recall_per_class[0]),
                "f1_score": float(f1_per_class[0]),
                "support": int(support_per_class[0]),
            },
            "positive": {
                "precision": float(precision_per_class[1]),
                "recall": float(recall_per_class[1]),
                "f1_score": float(f1_per_class[1]),
                "support": int(support_per_class[1]),
            },
        },
        "confusion_matrix": {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        },
    }

    return metrics


def print_evaluation_summary(metrics: Dict[str, Any]):
    """Print formatted evaluation summary"""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print("\nOverall Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("\nPer-Class Performance:")
    print("\n  Negative Class:")
    neg = metrics["per_class"]["negative"]
    print(f"    Precision: {neg['precision']:.4f}")
    print(f"    Recall:    {neg['recall']:.4f}")
    print(f"    F1-Score:  {neg['f1_score']:.4f}")
    print(f"    Support:   {neg['support']}")

    print("\n  Positive Class:")
    pos = metrics["per_class"]["positive"]
    print(f"    Precision: {pos['precision']:.4f}")
    print(f"    Recall:    {pos['recall']:.4f}")
    print(f"    F1-Score:  {pos['f1_score']:.4f}")
    print(f"    Support:   {pos['support']}")

    print("\nConfusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"                  Predicted")
    print(f"                  Negative  Positive")
    print(f"  Actual Negative   {cm['true_negative']:5d}     {cm['false_positive']:5d}")
    print(f"         Positive   {cm['false_negative']:5d}     {cm['true_positive']:5d}")

    print("\nAccuracy Breakdown:")
    total = (
        cm["true_negative"]
        + cm["false_positive"]
        + cm["false_negative"]
        + cm["true_positive"]
    )
    tn_rate = cm["true_negative"] / (cm["true_negative"] + cm["false_positive"]) * 100
    tp_rate = cm["true_positive"] / (cm["true_positive"] + cm["false_negative"]) * 100
    print(f"  True Negatives:  {cm['true_negative']:5d} ({tn_rate:.1f}% of negatives)")
    print(f"  True Positives:  {cm['true_positive']:5d} ({tp_rate:.1f}% of positives)")
    print(f"  False Positives: {cm['false_positive']:5d}")
    print(f"  False Negatives: {cm['false_negative']:5d}")

    print("\n" + "=" * 70)


def save_results(
    metrics: Dict[str, Any],
    true_labels: list,
    predicted_labels: list,
    probabilities: list,
    output_dir: str,
):
    """Save evaluation results to disk"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save metrics as JSON
    metrics_file = output_path / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")

    # Save classification report
    report = classification_report(
        true_labels, predicted_labels, target_names=["Negative", "Positive"]
    )
    report_file = output_path / "classification_report.txt"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Classification report saved to: {report_file}")

    # Save predictions for error analysis
    predictions_df = pd.DataFrame(
        {
            "true_label": true_labels,
            "predicted_label": predicted_labels,
            "prob_negative": [p[0] for p in probabilities],
            "prob_positive": [p[1] for p in probabilities],
            "correct": [t == p for t, p in zip(true_labels, predicted_labels)],
            "confidence": [max(p) for p in probabilities],
        }
    )
    predictions_file = output_path / "predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to: {predictions_file}")

    # Save summary statistics
    summary = {
        "total_samples": len(true_labels),
        "correct_predictions": sum(
            t == p for t, p in zip(true_labels, predicted_labels)
        ),
        "incorrect_predictions": sum(
            t != p for t, p in zip(true_labels, predicted_labels)
        ),
        "average_confidence": float(np.mean([max(p) for p in probabilities])),
        "median_confidence": float(np.median([max(p) for p in probabilities])),
    }
    summary_file = output_path / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")


def main():
    """Main evaluation pipeline"""
    args = parse_args()

    print("=" * 70)
    print("SENTIMENT ANALYSIS MODEL EVALUATION")
    print("=" * 70)
    print(f"\nModel path: {args.model_path}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.random_seed}")

    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"\nError: Model not found at {args.model_path}")
        print("Please train a model first or provide correct path.")
        sys.exit(1)

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    try:
        predictor = SentimentPredictor(args.model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"\nError loading model: {e}")
        sys.exit(1)

    # Load test data
    try:
        test_dataset = load_test_data(args.data_path, args.random_seed)
    except Exception as e:
        print(f"\nError loading test data: {e}")
        sys.exit(1)

    # Run predictions
    try:
        true_labels, predicted_labels, probabilities = predict_on_dataset(
            predictor, test_dataset
        )
    except Exception as e:
        print(f"\nError during prediction: {e}")
        sys.exit(1)

    # Calculate metrics
    try:
        metrics = calculate_metrics(true_labels, predicted_labels, probabilities)
    except Exception as e:
        print(f"\nError calculating metrics: {e}")
        sys.exit(1)

    # Print results
    print_evaluation_summary(metrics)

    # Save results
    try:
        save_results(
            metrics, true_labels, predicted_labels, probabilities, args.output_dir
        )
    except Exception as e:
        print(f"\nError saving results: {e}")
        sys.exit(1)

    print("\nEvaluation complete!")

    # Return success/failure based on performance
    if metrics["accuracy"] >= 0.90:
        print("\n✅ Model meets performance target (90%+ accuracy)")
        return 0
    else:
        print(
            f"\n⚠️  Model below target: {metrics['accuracy']*100:.1f}% < 90% accuracy"
        )
        print("Consider additional training or hyperparameter tuning.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
