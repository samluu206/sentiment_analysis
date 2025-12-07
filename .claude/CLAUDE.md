# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a production-ready BERT-based sentiment analysis system for product reviews. The project is structured as a professional Python package designed for Big Tech ML/AI Software Engineer portfolios, demonstrating end-to-end ML engineering from data collection to model deployment.

**Target**: Achieving 95%+ accuracy on Amazon review sentiment classification (positive/negative)

**Current Status**: Phase 3 in progress - Model training complete with 91.875% test accuracy (exceeds 90% milestone). Currently working on comprehensive testing and validation. Data pipeline complete, MLflow integrated, production model trained (3 epochs, 20K samples). See `PROJECT_STATUS.md` for detailed roadmap.

**Latest Milestone** (December 2025):
- ✅ Full 3-epoch BERT training completed
- ✅ Test Accuracy: 91.875% | F1: 0.922 | ROC-AUC: 0.971
- ✅ Evaluation infrastructure created (scripts/evaluate.py)
- ✅ Error analysis notebook implemented (notebooks/error_analysis.ipynb)

## Development Commands

### Environment Setup

**IMPORTANT**: This project uses a conda environment named `sentiment_analysis` located at:
`/home/luuchiquan/ENTER/envs/sentiment_analysis`

```bash
# Activate the conda environment (RECOMMENDED)
source ~/ENTER/etc/profile.d/conda.sh
conda activate sentiment_analysis

# Verify packages are available
python3 -c "import torch, transformers, pandas; print('Environment ready!')"
```

### GPU Setup (NVIDIA or AMD)

**Step 1: Install PyTorch (Choose Based on Your Hardware)**

**For NVIDIA GPUs (CUDA 11.8+):**
```bash
pip3 install torch torchvision torchaudio
```

**For AMD GPUs (ROCm 6.1):**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

**For CPU Only (No GPU):**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Step 2: Verify GPU Detection**
```bash
# Check if GPU is detected
python3 -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Expected output for AMD GPU:
# GPU Available: True
# Device: AMD Radeon RX 7900 XTX

# Expected output for NVIDIA GPU:
# GPU Available: True
# Device: NVIDIA GeForce RTX 4090
```

**Step 3: Install Other Dependencies**
```bash
pip install -r requirements.txt
```

**Hardware Requirements:**
- **NVIDIA GPUs**: RTX 3060+ (12GB+ VRAM), Tesla/A100 for production
- **AMD GPUs**:
  - RDNA 2: RX 6600M (8GB), RX 6800 (16GB), RX 6900 XT (16GB)
  - RDNA 3: RX 7900 XTX (24GB), RX 7900 XT (20GB)
  - Data Center: MI200+ series
- **CPU**: 16GB+ RAM (training will be 10-20x slower)

**Troubleshooting GPU Issues:**

If GPU is not detected after installing ROCm PyTorch:
```bash
# Check ROCm drivers
rocm-smi  # Should show your AMD GPU

# Add user to video/render groups (AMD GPUs)
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER
# Logout and login again

# Set environment variables (if needed for AMD)
# For RDNA 2 (RX 6600M, RX 6800, RX 6900 XT):
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# For RDNA 3 (RX 7900 XTX, RX 7900 XT):
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

Alternative (venv setup - only if conda environment is not available):
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install for development
pip install -e ".[dev]"
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Code Quality
```bash
# Format code
make format
# Or manually:
black src/ scripts/ tests/ --line-length=100
isort src/ scripts/ tests/ --profile=black

# Run linters
make lint
# Or manually:
flake8 src/ scripts/ tests/ --max-line-length=100 --ignore=E203,W503
mypy src/ --ignore-missing-imports

# Run all pre-commit hooks
pre-commit run --all-files
```

### Testing
```bash
# Run all tests with coverage
make test
# Or:
pytest tests/ -v --cov=src/sentiment_analyzer --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_data_validator.py -v

# Run specific test function
pytest tests/test_data_validator.py::TestDataValidator::test_validate_schema_valid -v

# Run with specific markers
pytest -m unit -v
pytest -m "not slow" -v
```

### Data Collection
```bash
# Collect 10K reviews from Hugging Face
python scripts/collect_data.py \
    --dataset amazon_polarity \
    --sample-size 10000 \
    --output-file amazon_10k.csv

# Collect multilingual reviews with categories
python scripts/collect_data.py \
    --dataset amazon_reviews_multi \
    --sample-size 20000 \
    --output-file amazon_multi.csv
```

### Training

**Note**: Always activate the conda environment first:
```bash
source ~/ENTER/etc/profile.d/conda.sh && conda activate sentiment_analysis
```

Then run training commands:
```bash
# Train with default settings
make train

# Train with MLflow experiment tracking
make train-mlflow

# Train with custom parameters
python3 scripts/train.py \
    --sample-size 10000 \
    --epochs 3 \
    --batch-size 16 \
    --learning-rate 2e-5 \
    --output-dir models/experiment_1

# Train with MLflow tracking and custom run name
python3 scripts/train.py \
    --sample-size 10000 \
    --epochs 3 \
    --use-mlflow \
    --run-name "bert_baseline_experiment"

# Train with 3-way split (60/20/20: train/val/test)
python3 scripts/train.py \
    --data-path data/raw/amazon_polarity_20k.csv \
    --sample-size 20000 \
    --epochs 3 \
    --batch-size 16
```

### MLflow Experiment Tracking
```bash
# Start MLflow UI to view experiments
make mlflow-ui

# Access at http://localhost:5000
# View all training runs, compare metrics, and download models
```

### Inference
```bash
# Interactive prediction mode
make predict

# Single prediction with confidence
python scripts/predict.py \
    --model-path models/final_sentiment_bert \
    --text "This product is amazing!" \
    --with-confidence
```

## Architecture Overview

### Package Structure Philosophy

The codebase follows a **modular src-layout** design pattern:
- `src/sentiment_analyzer/`: Main package (importable after pip install)
- `scripts/`: Executable entry points (training, inference, data collection)
- `tests/`: Test suite mirroring src/ structure
- `notebooks/`: Exploratory analysis and prototyping

### Core Pipeline Flow

```
Data Collection → Validation → Preprocessing → Training (+ MLflow) → Evaluation → Inference
     ↓              ↓             ↓              ↓                      ↓           ↓
amazon_api.py  validator.py  preprocessor.py  trainer.py             helpers.py  predictor.py
                                                  ↓
                                            Experiment Tracking
                                            (mlruns/ directory)
```

### Key Design Patterns

1. **Configuration Management** (`utils/config.py`):
   - Uses dataclasses for type safety
   - Hierarchical configs: `Config` → `DataConfig`, `ModelConfig`, `TrainingConfig`
   - Centralized defaults (e.g., train_split=0.6, test_split=0.2, max_length=128)

2. **Data Pipeline** (`data/`):
   - `data_loader.py`: Loads CSV, creates train/val/test splits (60/20/20, stratified, random sampling)
   - `preprocessor.py`: Tokenizes with Hugging Face tokenizers
   - `validator.py`: Schema + quality validation (text length, labels, duplicates)
   - `amazon_api.py`: Hugging Face dataset integration (NOT Amazon PA-API - doesn't provide review text)

3. **Model Management** (`models/`):
   - `model.py`: Wraps AutoModelForSequenceClassification with utilities
   - `trainer.py`: Encapsulates Hugging Face Trainer with custom metrics
   - Model: `nlptown/bert-base-multilingual-uncased-sentiment` (110M params)
   - Fine-tuned classification head: 5 labels → 2 labels (binary sentiment)

4. **Inference** (`inference/`):
   - `predictor.py`: Single + batch prediction with confidence scores
   - Supports both label-only and detailed probability outputs

5. **Experiment Tracking** (MLflow):
   - Integrated in `SentimentTrainer` for automatic logging
   - Tracks hyperparameters, metrics, and model artifacts
   - UI accessible via `make mlflow-ui` at http://localhost:5000
   - Storage: `mlruns/` directory (gitignored)

### Critical Implementation Details

**Configuration Defaults** (in `utils/config.py`):
- Max sequence length: **128 tokens** (good for product reviews)
- Train/Val/Test split: **60/20/20** (stratified by label, random sampling)
- Learning rate: **2e-5** (standard for BERT fine-tuning)
- Batch size: **16** (balanced for CPU/small GPU)
- Random seed: **42** (reproducibility)

**Data Schema Requirements**:
- Input CSV must have: `full_text` (review text) + `label` (0=negative, 1=positive)
- Text length constraints: 10-5000 characters (configurable in validator)
- Labels: Binary {0, 1} only

**Model I/O**:
- Saved models include: tokenizer, model weights (safetensors), config
- Models are ~640MB (not in git, use .gitignore)
- Training checkpoints saved every 50 steps in `results/`
- MLflow artifacts stored in `mlruns/` (experiment tracking data, model versions)

**Testing Strategy**:
- Unit tests for data validation and loading
- Target coverage: 80%+
- pytest markers: `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.unit`

## Important Context

### Amazon Product Advertising API Limitation
**Critical**: The Amazon Product Advertising API (PA-API) does NOT provide review text - only product metadata and ratings.

**Data Collection Strategy**: Use Hugging Face public datasets instead:
- `amazon_polarity`: 4M reviews, binary labels (recommended)
- `amazon_reviews_multi`: Multilingual with categories
- These are legal, compliant, and pre-labeled

The `amazon_api.py` module integrates with Hugging Face, not Amazon PA-API.

### Code Style Standards
- Line length: **100 characters** (black, flake8)
- Import sorting: **black profile** (isort)
- Type hints: Encouraged but not strictly enforced (mypy --ignore-missing-imports)
- Docstrings: Google style (Args, Returns, Raises)

### Pre-commit Hooks
All code must pass:
- `black`: Auto-formatting
- `isort`: Import organization
- `flake8`: Linting (ignore E203, W503 for black compatibility)
- `mypy`: Type checking (lenient mode)
- Standard hooks: trailing whitespace, end-of-file, YAML/JSON validation

## Programmatic Usage Examples

### Training Pipeline
```python
from sentiment_analyzer import (
    SentimentDataLoader, TextPreprocessor,
    SentimentModel, SentimentTrainer, Config
)

config = Config()

# Load data with 60/20/20 split
loader = SentimentDataLoader(
    data_path="data/raw/reviews.csv",
    train_split=0.6,
    test_split=0.2,
    random_seed=42
)
train_ds, val_ds, test_ds = loader.load_and_prepare()

# Preprocess
preprocessor = TextPreprocessor(
    model_name=config.model.model_name,
    max_length=128
)
train_ds = preprocessor.process_dataset(train_ds)
val_ds = preprocessor.process_dataset(val_ds)

# Train
model_wrapper = SentimentModel(
    model_name=config.model.model_name,
    num_labels=2
)
model = model_wrapper.load_model()

trainer_wrapper = SentimentTrainer(
    model=model,
    tokenizer=preprocessor.get_tokenizer(),
    num_epochs=3,
    batch_size=16
)
trainer = trainer_wrapper.train(train_ds, val_ds)

# Save
model_wrapper.save_model("models/my_model")
preprocessor.get_tokenizer().save_pretrained("models/my_model")
```

### Inference Pipeline
```python
from sentiment_analyzer import SentimentPredictor

predictor = SentimentPredictor(model_path="models/final_sentiment_bert")

# Single prediction with confidence
result = predictor.predict_with_confidence(
    "This product exceeded my expectations!"
)
print(f"{result['sentiment']} ({result['confidence']:.2%})")

# Batch prediction (efficient for large datasets)
texts = ["Great!", "Terrible.", "It's okay."]
results = predictor.batch_predict(texts, batch_size=32)
```

### Data Validation
```python
from sentiment_analyzer.data.validator import DataValidator
import pandas as pd

df = pd.read_csv("data/raw/reviews.csv")

validator = DataValidator(
    text_column="full_text",
    label_column="label",
    min_text_length=10,
    max_text_length=5000
)

# Full validation with report
df_clean, report = validator.validate_dataset(df, verbose=True)

# Check for duplicates
df_clean, dup_count = validator.detect_duplicates(df_clean, remove=True)

df_clean.to_csv("data/processed/cleaned.csv", index=False)
```

## Common Workflows

### Adding a New Model Architecture
1. Create new config in `utils/config.py` (e.g., `DistilBERTConfig`)
2. Extend `SentimentModel` or create new model class in `models/model.py`
3. Update `scripts/train.py` to support new model via CLI argument
4. Add tests in `tests/test_models.py`
5. Document in README and PROJECT_STATUS

### Expanding the API (Future)
1. Implement endpoints in `src/sentiment_analyzer/api/`
2. Use FastAPI + Pydantic for validation
3. Add API tests in `tests/test_api.py`
4. Create Dockerfile for containerization
5. Update deployment documentation

### Using MLflow Experiment Tracking
1. Start MLflow UI: `make mlflow-ui` (access at http://localhost:5000)
2. Train with tracking: `make train-mlflow` or add `--use-mlflow` flag
3. Compare experiments in UI: view metrics, parameters, and model artifacts
4. Access logged data programmatically:
   ```python
   import mlflow

   # Search for best run
   runs = mlflow.search_runs(experiment_ids=["0"])
   best_run = runs.loc[runs["metrics.accuracy"].idxmax()]

   # Load best model
   model = mlflow.pytorch.load_model(f"runs:/{best_run.run_id}/model")
   ```
5. MLflow automatically logs: hyperparameters, metrics (accuracy, F1, loss), and model artifacts

## Project Roadmap Context

This project is **Week 3-4 of an 8-week plan** to create a portfolio-quality ML project:
- **Completed**:
  - ✅ Project restructuring
  - ✅ Data pipeline (60/20/20 split with random sampling)
  - ✅ Validation and testing foundation
  - ✅ MLflow integration
  - ✅ Production BERT model (91.875% test accuracy, exceeds 90% target)
  - ✅ Full 3-epoch training on 20K samples
  - ✅ Evaluation infrastructure (scripts/evaluate.py)
  - ✅ Error analysis notebook (notebooks/error_analysis.ipynb)
- **Current Phase**: Comprehensive testing, error analysis, performance benchmarking, test coverage to 80%+
- **Next Phase**: Model comparison experiments (BERT vs DistilBERT vs RoBERTa), FastAPI deployment, Docker, CI/CD
- **Final Phase**: Web demo, monitoring, documentation finalization, portfolio presentation

See `PROJECT_STATUS.md` and `docs/PROJECT_ASSESSMENT.md` for complete roadmap and phase details.

## Key Files Reference

- `setup.py`: Package metadata and dependencies
- `Makefile`: Common development commands (train, test, mlflow-ui, etc.)
- `pytest.ini`: Test configuration (coverage, markers)
- `.pre-commit-config.yaml`: Code quality hooks
- `src/sentiment_analyzer/utils/config.py`: All configuration defaults
- `scripts/start_mlflow.py`: MLflow UI server launcher
- `scripts/train.py`: Training script with MLflow integration
- `scripts/evaluate.py`: Comprehensive model evaluation script
- `notebooks/error_analysis.ipynb`: Error analysis and visualization notebook
- `docs/DATA_COLLECTION.md`: Complete data collection guide
- `docs/PROJECT_ASSESSMENT.md`: FAANG-level project assessment
- `PROJECT_STATUS.md`: Detailed project status and roadmap
- `mlruns/`: MLflow experiment tracking data (gitignored)
- `models/trained_model_v2/`: Latest production model (91.875% accuracy, gitignored)
