"""MLflow experiment tracking utilities."""
import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Manages MLflow experiment tracking for sentiment analysis."""

    def __init__(
        self,
        experiment_name: str = "sentiment-analysis",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local file storage)
            artifact_location: Location to store artifacts
        """
        self.experiment_name = experiment_name

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri("file:./mlruns")

        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )
                logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

            mlflow.set_experiment(experiment_name)
            self.experiment_id = experiment_id

        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {e}")
            raise

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None):
        """
        Start a new MLflow run.

        Args:
            run_name: Optional name for the run
            tags: Optional tags to attach to the run
        """
        mlflow.start_run(run_name=run_name)

        if tags:
            mlflow.set_tags(tags)

        logger.info(f"Started MLflow run: {mlflow.active_run().info.run_id}")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.

        Args:
            params: Dictionary of parameters to log
        """
        try:
            mlflow.log_params(params)
            logger.debug(f"Logged {len(params)} parameters")
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged {len(metrics)} metrics at step {step}")
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")

    def log_model(
        self,
        model,
        artifact_path: str = "model",
        **kwargs
    ):
        """
        Log PyTorch model to MLflow.

        Args:
            model: PyTorch model to log
            artifact_path: Path within the run's artifact directory
            **kwargs: Additional arguments for mlflow.pytorch.log_model
        """
        try:
            mlflow.pytorch.log_model(model, artifact_path, **kwargs)
            logger.info(f"Logged model to {artifact_path}")
        except Exception as e:
            logger.error(f"Error logging model: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact file to MLflow.

        Args:
            local_path: Local file path
            artifact_path: Optional subdirectory in artifact store
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"Error logging artifact: {e}")

    def log_figure(self, figure, artifact_file: str):
        """
        Log a matplotlib figure to MLflow.

        Args:
            figure: Matplotlib figure object
            artifact_file: Filename for the artifact
        """
        try:
            mlflow.log_figure(figure, artifact_file)
            logger.debug(f"Logged figure: {artifact_file}")
        except Exception as e:
            logger.error(f"Error logging figure: {e}")

    def log_dict(self, dictionary: Dict, artifact_file: str):
        """
        Log a dictionary as JSON artifact.

        Args:
            dictionary: Dictionary to log
            artifact_file: Filename for the JSON artifact
        """
        try:
            mlflow.log_dict(dictionary, artifact_file)
            logger.debug(f"Logged dictionary: {artifact_file}")
        except Exception as e:
            logger.error(f"Error logging dictionary: {e}")

    def log_text(self, text: str, artifact_file: str):
        """
        Log text content as artifact.

        Args:
            text: Text content
            artifact_file: Filename for the text artifact
        """
        try:
            mlflow.log_text(text, artifact_file)
            logger.debug(f"Logged text: {artifact_file}")
        except Exception as e:
            logger.error(f"Error logging text: {e}")

    def set_tags(self, tags: Dict[str, str]):
        """
        Set tags for the current run.

        Args:
            tags: Dictionary of tags
        """
        try:
            mlflow.set_tags(tags)
            logger.debug(f"Set {len(tags)} tags")
        except Exception as e:
            logger.error(f"Error setting tags: {e}")

    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        try:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run with status: {status}")
        except Exception as e:
            logger.error(f"Error ending run: {e}")

    @staticmethod
    def get_run_id() -> Optional[str]:
        """Get the current run ID."""
        active_run = mlflow.active_run()
        return active_run.info.run_id if active_run else None

    @staticmethod
    def load_model(model_uri: str):
        """
        Load a model from MLflow.

        Args:
            model_uri: MLflow model URI

        Returns:
            Loaded model
        """
        try:
            return mlflow.pytorch.load_model(model_uri)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


def get_run_metrics(run_id: str) -> Dict[str, float]:
    """
    Get all metrics for a specific run.

    Args:
        run_id: MLflow run ID

    Returns:
        Dictionary of metric names and values
    """
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return run.data.metrics


def compare_runs(run_ids: list, metric_name: str) -> Dict[str, float]:
    """
    Compare a specific metric across multiple runs.

    Args:
        run_ids: List of run IDs to compare
        metric_name: Name of the metric to compare

    Returns:
        Dictionary mapping run_id to metric value
    """
    client = mlflow.tracking.MlflowClient()
    results = {}

    for run_id in run_ids:
        run = client.get_run(run_id)
        if metric_name in run.data.metrics:
            results[run_id] = run.data.metrics[metric_name]

    return results


def get_best_run(experiment_id: str, metric_name: str, ascending: bool = False) -> Optional[str]:
    """
    Get the best run from an experiment based on a metric.

    Args:
        experiment_id: MLflow experiment ID
        metric_name: Metric to optimize
        ascending: True if lower is better (e.g., loss)

    Returns:
        Run ID of the best run
    """
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
        max_results=1
    )

    return runs[0].info.run_id if runs else None
