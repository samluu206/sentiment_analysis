import pytest
import tempfile
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.sentiment_analyzer.data.data_loader import SentimentDataLoader
from src.sentiment_analyzer.data.preprocessor import TextPreprocessor
from src.sentiment_analyzer.models.model import SentimentModel
from src.sentiment_analyzer.inference.predictor import SentimentPredictor


@pytest.fixture(scope="module")
def sample_data_file():
    """Create a temporary CSV file with sample data."""
    tmpdir = tempfile.mkdtemp()
    file_path = Path(tmpdir) / "sample_data.csv"

    data = {
        "full_text": [
            "This product is amazing! Highly recommend.",
            "Terrible quality. Waste of money.",
            "Average product, nothing special.",
            "Excellent customer service and fast delivery.",
            "Poor packaging, item arrived damaged.",
            "Good value for money.",
            "Not as described, very disappointing.",
            "Perfect! Exactly what I needed.",
            "Mediocre quality, expected better.",
            "Outstanding product, will buy again!"
        ],
        "label": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    }

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

    yield file_path

    import shutil
    shutil.rmtree(tmpdir)


@pytest.fixture(scope="module")
def trained_model_dir():
    """Create a temporary directory with a small trained model."""
    tmpdir = tempfile.mkdtemp()
    model_path = Path(tmpdir)

    model_name = "prajjwal1/bert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path, safe_serialization=True)

    yield model_path

    import shutil
    shutil.rmtree(tmpdir)


class TestEndToEndPipeline:
    """Integration tests for the complete pipeline."""

    def test_full_training_pipeline(self, sample_data_file):
        """Test complete training pipeline from data to model."""
        loader = SentimentDataLoader(
            data_path=str(sample_data_file),
            train_split=0.6,
            test_split=0.2,
            random_seed=42
        )

        train_ds, val_ds, test_ds, _ = loader.load_and_prepare_with_test()

        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0

        total_samples = len(train_ds) + len(val_ds) + len(test_ds)
        assert total_samples == 10

        preprocessor = TextPreprocessor(
            model_name="prajjwal1/bert-tiny",
            max_length=128
        )

        train_tokenized = preprocessor.process_dataset(train_ds)

        assert "input_ids" in train_tokenized.column_names
        assert "attention_mask" in train_tokenized.column_names
        assert "label" in train_tokenized.column_names

    def test_full_inference_pipeline(self, sample_data_file, trained_model_dir):
        """Test complete inference pipeline from data to prediction."""
        predictor = SentimentPredictor(
            model_path=str(trained_model_dir),
            device="cpu"
        )

        df = pd.read_csv(sample_data_file)
        texts = df["full_text"].tolist()

        results = predictor.batch_predict(texts, batch_size=4)

        assert len(results) == len(texts)

        for result in results:
            assert "sentiment" in result
            assert "confidence" in result
            assert result["sentiment"] in ["POSITIVE", "NEGATIVE"]
            assert 0.0 <= result["confidence"] <= 1.0

    def test_model_save_and_load_cycle(self):
        """Test saving and loading a model."""
        model_name = "prajjwal1/bert-tiny"

        model1 = SentimentModel(
            model_name=model_name,
            num_labels=2,
            device="cpu"
        )
        model1.load_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            model1.save_model(str(save_path))

            predictor = SentimentPredictor(
                model_path=str(save_path),
                device="cpu"
            )

            result = predictor.predict("This is a test.")

            assert result in ["POSITIVE", "NEGATIVE"]

    def test_data_pipeline_with_preprocessing(self, sample_data_file):
        """Test data loading and preprocessing together."""
        loader = SentimentDataLoader(
            data_path=str(sample_data_file),
            train_split=0.7,
            val_split=0.1,
            test_split=0.2,
            random_seed=42
        )

        train_ds, val_ds, _, _ = loader.load_and_prepare_with_test()

        preprocessor = TextPreprocessor(
            model_name="prajjwal1/bert-tiny",
            max_length=64
        )

        train_processed = preprocessor.process_dataset(train_ds)
        val_processed = preprocessor.process_dataset(val_ds)

        assert "input_ids" in train_processed.column_names
        assert "input_ids" in val_processed.column_names

        for i in range(len(train_processed)):
            assert len(train_processed[i]["input_ids"]) <= 64

    def test_prediction_consistency_across_formats(self, trained_model_dir):
        """Test that predictions are consistent across different input formats."""
        predictor = SentimentPredictor(
            model_path=str(trained_model_dir),
            device="cpu"
        )

        text = "This is a great product!"

        single_result = predictor.predict(text)

        batch_result = predictor.predict([text])

        confidence_result = predictor.predict_with_confidence(text)

        batch_predict_result = predictor.batch_predict([text])

        assert single_result == batch_result[0]
        assert single_result == confidence_result["sentiment"]
        assert single_result == batch_predict_result[0]["sentiment"]

    def test_model_parameters_counted_correctly(self):
        """Test that model parameter counting works."""
        model = SentimentModel(
            model_name="prajjwal1/bert-tiny",
            num_labels=2,
            device="cpu"
        )

        total_params = model.get_num_parameters()
        trainable_params = model.get_trainable_parameters()

        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params

    def test_data_splits_sum_correctly(self, sample_data_file):
        """Test that data splits sum to total dataset size."""
        df = pd.read_csv(sample_data_file)
        total_rows = len(df)

        loader = SentimentDataLoader(
            data_path=str(sample_data_file),
            train_split=0.5,
            test_split=0.3,
            random_seed=42
        )

        train_ds, val_ds, test_ds, _ = loader.load_and_prepare_with_test()

        assert len(train_ds) + len(val_ds) + len(test_ds) == total_rows

    def test_stratified_sampling_preserves_distribution(self, sample_data_file):
        """Test that stratified sampling preserves class distribution."""
        df = pd.read_csv(sample_data_file)
        original_positive_ratio = df["label"].mean()

        loader = SentimentDataLoader(
            data_path=str(sample_data_file),
            train_split=0.6,
            test_split=0.2,
            random_seed=42
        )

        train_ds, val_ds, test_ds, _ = loader.load_and_prepare_with_test()

        train_labels = train_ds["label"]
        val_labels = val_ds["label"]
        test_labels = test_ds["label"]

        train_positive_ratio = sum(train_labels) / len(train_labels)
        val_positive_ratio = sum(val_labels) / len(val_labels) if len(val_labels) > 0 else 0
        test_positive_ratio = sum(test_labels) / len(test_labels) if len(test_labels) > 0 else 0

        assert abs(train_positive_ratio - original_positive_ratio) < 0.3
        if len(val_labels) > 0:
            assert abs(val_positive_ratio - original_positive_ratio) < 0.5
        if len(test_labels) > 0:
            assert abs(test_positive_ratio - original_positive_ratio) < 0.5

    def test_tokenizer_compatibility_across_components(self):
        """Test that tokenizer works consistently across components."""
        model_name = "prajjwal1/bert-tiny"

        preprocessor = TextPreprocessor(model_name=model_name, max_length=128)

        model = SentimentModel(model_name=model_name, num_labels=2, device="cpu")
        model.load_model()

        text = "Test sentence for tokenization"
        data = {"full_text": [text]}

        tokenized = preprocessor.tokenize_function(data)

        assert "input_ids" in tokenized
        assert len(tokenized["input_ids"][0]) > 0

    def test_reproducibility_with_random_seed(self, sample_data_file):
        """Test that results are reproducible with same random seed."""
        loader1 = SentimentDataLoader(
            data_path=str(sample_data_file),
            train_split=0.6,
            test_split=0.2,
            random_seed=42
        )

        train_ds1, val_ds1, test_ds1, _ = loader1.load_and_prepare_with_test()

        loader2 = SentimentDataLoader(
            data_path=str(sample_data_file),
            train_split=0.6,
            test_split=0.2,
            random_seed=42
        )

        train_ds2, val_ds2, test_ds2, _ = loader2.load_and_prepare_with_test()

        assert train_ds1["label"] == train_ds2["label"]
        assert val_ds1["label"] == val_ds2["label"]
        assert test_ds1["label"] == test_ds2["label"]

    def test_prediction_on_preprocessed_data(self, sample_data_file, trained_model_dir):
        """Test making predictions on preprocessed dataset."""
        loader = SentimentDataLoader(
            data_path=str(sample_data_file),
            train_split=0.5,
            val_split=0.1,
            test_split=0.4,
            random_seed=42
        )

        _, _, test_ds, _ = loader.load_and_prepare_with_test()

        predictor = SentimentPredictor(
            model_path=str(trained_model_dir),
            device="cpu"
        )

        texts = test_ds["full_text"]
        predictions = predictor.predict(texts)

        assert len(predictions) == len(texts)
        assert all(pred in ["POSITIVE", "NEGATIVE"] for pred in predictions)

    def test_error_handling_in_pipeline(self):
        """Test error handling in the pipeline."""
        with pytest.raises(Exception):
            SentimentPredictor(model_path="/nonexistent/path", device="cpu")

    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_file = Path(tmpdir) / "empty.csv"
            pd.DataFrame({"full_text": [], "label": []}).to_csv(empty_file, index=False)

            loader = SentimentDataLoader(
                data_path=str(empty_file),
                train_split=0.6,
                test_split=0.2,
                random_seed=42
            )

            with pytest.raises(Exception):
                loader.load_and_prepare_with_test()
