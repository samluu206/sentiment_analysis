"""Training script for sentiment analysis model."""
import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentiment_analyzer import (
    SentimentDataLoader,
    TextPreprocessor,
    SentimentModel,
    SentimentTrainer,
    Config
)
from sentiment_analyzer.utils.helpers import setup_logging, save_metrics
from sentiment_analyzer.utils.mlflow_tracker import MLflowTracker


def main(args):
    """Main training function."""
    config = Config()

    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting sentiment analysis training")

    mlflow_tracker = None
    if args.use_mlflow:
        logger.info("Initializing MLflow tracking")
        mlflow_tracker = MLflowTracker(
            experiment_name=args.experiment_name,
            tracking_uri=args.mlflow_tracking_uri
        )
        mlflow_tracker.start_run(
            run_name=args.run_name,
            tags={
                "model_type": "BERT",
                "task": "sentiment_classification",
                "dataset": "amazon_reviews"
            }
        )

    data_path = Path(config.project_root) / "data" / "raw" / "amazon_polarity_sample.csv"
    logger.info(f"Loading data from {data_path}")

    data_loader = SentimentDataLoader(
        data_path=str(data_path),
        train_split=config.data.train_split,
        random_seed=config.data.random_seed
    )

    train_dataset, val_dataset, df = data_loader.load_and_prepare(
        sample_size=args.sample_size
    )
    logger.info(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")

    logger.info("Initializing preprocessor")
    preprocessor = TextPreprocessor(
        model_name=config.model.model_name,
        max_length=config.model.max_length
    )

    logger.info("Tokenizing datasets")
    train_dataset = preprocessor.process_dataset(train_dataset)
    val_dataset = preprocessor.process_dataset(val_dataset)

    logger.info("Loading model")
    model_wrapper = SentimentModel(
        model_name=config.model.model_name,
        num_labels=config.model.num_labels
    )
    model = model_wrapper.load_model()
    logger.info(f"Model has {model_wrapper.get_num_parameters():,} parameters")

    logger.info("Initializing trainer")
    trainer_wrapper = SentimentTrainer(
        model=model,
        tokenizer=preprocessor.get_tokenizer(),
        output_dir=config.training.output_dir,
        num_epochs=args.epochs or config.training.num_epochs,
        batch_size=args.batch_size or config.training.batch_size,
        learning_rate=args.learning_rate or config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        use_mlflow=args.use_mlflow,
        mlflow_tracker=mlflow_tracker
    )

    try:
        logger.info("Starting training")
        trainer = trainer_wrapper.train(train_dataset, val_dataset)

        logger.info("Evaluating model")
        eval_results = trainer_wrapper.evaluate(trainer, val_dataset)
        logger.info(f"Evaluation results: {eval_results}")

        output_path = Path(config.project_root) / args.output_dir
        logger.info(f"Saving model to {output_path}")
        model_wrapper.save_model(str(output_path))

        tokenizer = preprocessor.get_tokenizer()
        tokenizer.save_pretrained(str(output_path))

        metrics_path = output_path / "metrics.json"
        save_metrics(eval_results, str(metrics_path))
        logger.info(f"Saved metrics to {metrics_path}")

        if args.use_mlflow:
            logger.info("Logging model to MLflow")
            mlflow_tracker.log_artifact(str(metrics_path), "metrics")
            mlflow_tracker.log_model(model, artifact_path="model")
            mlflow_tracker.end_run(status="FINISHED")

        logger.info("Training complete")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.use_mlflow and mlflow_tracker:
            mlflow_tracker.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of samples to use")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="models/trained_model", help="Output directory for model")

    parser.add_argument("--use-mlflow", action="store_true", help="Enable MLflow experiment tracking")
    parser.add_argument("--experiment-name", type=str, default="sentiment-analysis", help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking server URI")

    args = parser.parse_args()
    main(args)
