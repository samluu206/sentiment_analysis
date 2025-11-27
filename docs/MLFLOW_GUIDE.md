# MLflow Experiment Tracking Guide

This guide explains how to use MLflow for experiment tracking in the sentiment analyzer project. **Perfect for beginners!**

## What is MLflow?

MLflow is an open-source platform for managing the ML lifecycle, including:
- **Experiment tracking**: Log parameters, metrics, and models
- **Model registry**: Store and version models
- **Comparison**: Compare runs side-by-side

Think of it as a lab notebook for your ML experiments that automatically tracks everything.

## Quick Start

### 1. Install Dependencies

MLflow is already in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

### 2. Start MLflow UI

Open a terminal and run:

```bash
make mlflow-ui
# Or: python scripts/start_mlflow.py
```

This starts a local web server at: **http://localhost:5000**

Open this URL in your browser to see the MLflow dashboard.

### 3. Run Training with MLflow

In a **new terminal** (keep MLflow UI running), train a model:

```bash
make train-mlflow
# Or: python scripts/train.py --use-mlflow --sample-size 1000 --epochs 3
```

### 4. View Results

Go to http://localhost:5000 and you'll see:
- Your experiment run
- All logged parameters (learning rate, batch size, etc.)
- All metrics (accuracy, F1, precision, recall, ROC-AUC)
- The trained model

## Understanding MLflow Concepts

### Experiments
- A **project** or **question** you're investigating
- Example: "sentiment-analysis"
- Contains multiple runs

### Runs
- A **single training execution**
- Example: "baseline_bert_lr2e-5"
- Contains parameters, metrics, and artifacts

### Parameters
- **Input values** that don't change during training
- Examples: learning_rate=2e-5, batch_size=16

### Metrics
- **Output values** that change during training
- Examples: accuracy=0.95, f1=0.94

### Artifacts
- **Files** produced by the run
- Examples: model weights, plots, configs

## Usage Examples

### Example 1: Basic Training with MLflow

```bash
python scripts/train.py \
    --use-mlflow \
    --sample-size 5000 \
    --epochs 3 \
    --run-name "bert_5k_samples"
```

What gets logged:
- Parameters: num_epochs=3, batch_size=16, learning_rate=2e-5, train_samples=5000
- Metrics: accuracy, f1, precision, recall, roc_auc
- Artifacts: trained model, metrics.json

### Example 2: Compare Learning Rates

Run multiple experiments with different learning rates:

```bash
# Experiment 1
python scripts/train.py --use-mlflow --learning-rate 1e-5 --run-name "lr_1e-5"

# Experiment 2
python scripts/train.py --use-mlflow --learning-rate 2e-5 --run-name "lr_2e-5"

# Experiment 3
python scripts/train.py --use-mlflow --learning-rate 5e-5 --run-name "lr_5e-5"
```

Then in MLflow UI:
1. Select all 3 runs
2. Click "Compare"
3. See which learning rate gives best results

### Example 3: Different Batch Sizes

```bash
# Small batch
python scripts/train.py --use-mlflow --batch-size 8 --run-name "batch_8"

# Medium batch
python scripts/train.py --use-mlflow --batch-size 16 --run-name "batch_16"

# Large batch
python scripts/train.py --use-mlflow --batch-size 32 --run-name "batch_32"
```

### Example 4: Different Dataset Sizes

```bash
# 1K samples
python scripts/train.py --use-mlflow --sample-size 1000 --run-name "data_1k"

# 5K samples
python scripts/train.py --use-mlflow --sample-size 5000 --run-name "data_5k"

# 10K samples
python scripts/train.py --use-mlflow --sample-size 10000 --run-name "data_10k"
```

## MLflow UI Navigation

### Main Dashboard
- **Experiments**: List of all experiments (left sidebar)
- **Runs**: Table showing all runs for selected experiment
- **Columns**: run name, metrics, parameters, start time

### Run Detail Page
Click any run to see:
- **Overview**: Run ID, duration, status
- **Parameters**: All hyperparameters
- **Metrics**: All evaluation metrics with history
- **Artifacts**: Saved files (model, plots, configs)
- **Tags**: Metadata about the run

### Comparing Runs
1. Select multiple runs (checkboxes)
2. Click "Compare" button
3. See side-by-side comparison of:
   - Parameter differences
   - Metric comparisons (bar charts, scatter plots)
   - Best/worst runs

## Tips for Good Experiment Tracking

### 1. Use Descriptive Run Names

❌ Bad:
```bash
--run-name "test1"
--run-name "experiment"
```

✅ Good:
```bash
--run-name "bert_base_lr2e-5_batch16"
--run-name "distilbert_5k_samples"
```

### 2. Track Everything Important

Current implementation tracks:
- ✅ Hyperparameters (lr, batch size, epochs)
- ✅ Dataset size (train/val samples)
- ✅ Metrics (accuracy, F1, precision, recall, ROC-AUC)
- ✅ Model architecture (model name)
- ✅ Trained model

### 3. Organize Experiments

Create separate experiments for:
- Different models (BERT, DistilBERT, RoBERTa)
- Different tasks (binary classification, multi-class)
- Different datasets (amazon_polarity, custom data)

```bash
--experiment-name "bert-experiments"
--experiment-name "distilbert-experiments"
```

### 4. Add Notes to Runs

After training, click run → Edit → Add description:
- What was the goal?
- What did you learn?
- Any issues encountered?

## Advanced Usage

### Custom Experiment Name

```bash
python scripts/train.py \
    --use-mlflow \
    --experiment-name "hyperparameter-search" \
    --run-name "run_001"
```

### Programmatic Usage

```python
from sentiment_analyzer.utils.mlflow_tracker import MLflowTracker

# Initialize tracker
tracker = MLflowTracker(experiment_name="my-experiment")

# Start run
tracker.start_run(run_name="test_run", tags={"version": "v1"})

# Log parameters
tracker.log_params({"learning_rate": 2e-5, "batch_size": 16})

# Log metrics
tracker.log_metrics({"accuracy": 0.95, "f1": 0.94})

# Log model
tracker.log_model(model, artifact_path="model")

# End run
tracker.end_run(status="FINISHED")
```

### Query Best Run

```python
from sentiment_analyzer.utils.mlflow_tracker import get_best_run

# Get run with highest F1 score
best_run_id = get_best_run(
    experiment_id="1",
    metric_name="f1",
    ascending=False  # Higher is better
)

print(f"Best run: {best_run_id}")
```

## Troubleshooting

### Issue: MLflow UI shows "No experiments"

**Solution**: Run a training command with `--use-mlflow` first:
```bash
make train-mlflow
```

### Issue: Port 5000 already in use

**Solution**: Use a different port:
```bash
python scripts/start_mlflow.py --port 5001
```

### Issue: Can't see my run

**Solution**:
1. Check you used `--use-mlflow` flag
2. Refresh MLflow UI page
3. Check correct experiment is selected (left sidebar)

### Issue: Metrics not showing

**Solution**:
- Metrics are logged at the END of training
- Wait for training to complete
- Check training didn't error out

## MLflow Directory Structure

```
project/
├── mlruns/                  # MLflow tracking data
│   ├── 0/                   # Default experiment
│   ├── 1/                   # Your experiments
│   │   ├── <run-id-1>/     # Individual run
│   │   │   ├── meta.yaml   # Run metadata
│   │   │   ├── metrics/    # Metric values
│   │   │   ├── params/     # Parameters
│   │   │   └── artifacts/  # Model, plots, etc.
│   │   └── <run-id-2>/
│   └── models/              # Model registry (future)
```

This directory is in `.gitignore` (don't commit 640MB models!)

## Best Practices for Portfolio

### 1. Run Systematic Experiments

Show you know how to tune hyperparameters:
- 3-5 different learning rates
- 3 different batch sizes
- 3 different dataset sizes

### 2. Document Findings

After experiments, create a summary:
- Which configuration performed best?
- What did you learn?
- What would you try next?

### 3. Take Screenshots

Capture MLflow UI screenshots for:
- Comparison of multiple runs
- Best model metrics
- Parameter importance

Include these in your README or portfolio.

### 4. Export Results

From MLflow UI:
- Download run CSV (all runs)
- Export comparison charts
- Save for documentation

## Next Steps

1. **Run baseline experiment** with current setup
2. **Try 3-5 variations** of hyperparameters
3. **Document best configuration** in notes
4. **Screenshot results** for portfolio
5. **Move to Week 4**: Multi-model comparison (BERT vs DistilBERT vs RoBERTa)

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Quickstart](https://mlflow.org/docs/latest/quickstart.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)

## Summary

MLflow helps you:
- ✅ Never lose experiment results
- ✅ Compare models systematically
- ✅ Reproduce successful experiments
- ✅ Show organized work to recruiters

**Key command to remember**:
```bash
# Start UI (keep running)
make mlflow-ui

# Train with tracking (new terminal)
make train-mlflow
```

Happy experimenting! 🚀
