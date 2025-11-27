.PHONY: help install install-dev setup test lint format clean train predict mlflow-ui train-mlflow

help:
	@echo "Available commands:"
	@echo "  make install      - Install package dependencies"
	@echo "  make install-dev  - Install package with development dependencies"
	@echo "  make setup        - Complete project setup (venv + install + pre-commit)"
	@echo "  make test         - Run tests with pytest"
	@echo "  make lint         - Run linters (flake8, mypy)"
	@echo "  make format       - Format code with black and isort"
	@echo "  make clean        - Remove build artifacts and cache"
	@echo "  make train        - Run training script (without MLflow)"
	@echo "  make train-mlflow - Run training with MLflow tracking"
	@echo "  make mlflow-ui    - Start MLflow UI server"
	@echo "  make predict      - Run prediction in interactive mode"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install -r requirements-dev.txt

setup:
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "  source venv/bin/activate  (Linux/Mac)"
	@echo "  venv\\Scripts\\activate     (Windows)"

test:
	pytest tests/ -v --cov=src/sentiment_analyzer --cov-report=html --cov-report=term

lint:
	flake8 src/ scripts/ tests/ --max-line-length=100 --ignore=E203,W503
	mypy src/ --ignore-missing-imports

format:
	black src/ scripts/ tests/ --line-length=100
	isort src/ scripts/ tests/ --profile=black

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/

train:
	python scripts/train.py --sample-size 1000 --epochs 3 --output-dir models/trained_model

train-mlflow:
	python scripts/train.py --sample-size 1000 --epochs 3 --output-dir models/trained_model --use-mlflow --run-name "baseline_experiment"

mlflow-ui:
	python scripts/start_mlflow.py

predict:
	python scripts/predict.py --model-path models/final_sentiment_bert --interactive
