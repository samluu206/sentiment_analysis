import pytest
import torch
import tempfile
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.sentiment_analyzer.inference.predictor import SentimentPredictor


@pytest.fixture(scope="module")
def temp_model_dir():
    """Create a temporary model directory with a small trained model."""
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


class TestSentimentPredictor:
    """Tests for SentimentPredictor class."""

    def test_predictor_initialization(self, temp_model_dir):
        """Test predictor initialization."""
        predictor = SentimentPredictor(model_path=str(temp_model_dir), device="cpu")

        assert predictor.model is not None
        assert predictor.tokenizer is not None
        assert predictor.device == "cpu"
        assert predictor.label_map == {0: "NEGATIVE", 1: "POSITIVE"}

    def test_predict_single_text(self, temp_model_dir):
        """Test prediction on single text."""
        predictor = SentimentPredictor(model_path=str(temp_model_dir), device="cpu")

        result = predictor.predict("This is a great product!")

        assert isinstance(result, str)
        assert result in ["POSITIVE", "NEGATIVE"]

    def test_predict_multiple_texts(self, temp_model_dir):
        """Test prediction on multiple texts."""
        predictor = SentimentPredictor(model_path=str(temp_model_dir), device="cpu")

        texts = [
            "Great product!",
            "Terrible experience.",
            "It's okay."
        ]

        results = predictor.predict(texts)

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(r in ["POSITIVE", "NEGATIVE"] for r in results)

    def test_predict_with_probabilities(self, temp_model_dir):
        """Test prediction with probability output."""
        predictor = SentimentPredictor(model_path=str(temp_model_dir), device="cpu")

        result = predictor.predict(
            "This is a test.",
            return_probabilities=True
        )

        assert isinstance(result, dict)
        assert "predictions" in result
        assert "probabilities" in result
        assert "confidence" in result

        assert isinstance(result["predictions"], str)
        assert isinstance(result["probabilities"], list)
        assert len(result["probabilities"]) == 2
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_batch_with_probabilities(self, temp_model_dir):
        """Test batch prediction with probabilities."""
        predictor = SentimentPredictor(model_path=str(temp_model_dir), device="cpu")

        texts = ["Text 1", "Text 2"]
        result = predictor.predict(texts, return_probabilities=True)

        assert isinstance(result, dict)
        assert isinstance(result["predictions"], list)
        assert len(result["predictions"]) == 2
        assert len(result["probabilities"]) == 2
        assert len(result["confidence"]) == 2

    def test_predict_with_confidence(self, temp_model_dir):
        """Test predict_with_confidence method."""
        predictor = SentimentPredictor(model_path=str(temp_model_dir), device="cpu")

        result = predictor.predict_with_confidence("Great product!")

        assert isinstance(result, dict)
        assert "sentiment" in result
        assert "confidence" in result
        assert "negative_prob" in result
        assert "positive_prob" in result

        assert result["sentiment"] in ["POSITIVE", "NEGATIVE"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0.0 <= result["negative_prob"] <= 1.0
        assert 0.0 <= result["positive_prob"] <= 1.0

        assert abs(result["negative_prob"] + result["positive_prob"] - 1.0) < 0.01

    def test_batch_predict(self, temp_model_dir):
        """Test batch_predict method."""
        predictor = SentimentPredictor(model_path=str(temp_model_dir), device="cpu")

        texts = [
            "Excellent!",
            "Poor quality.",
            "Average product.",
            "Highly recommend!",
            "Waste of money."
        ]

        results = predictor.batch_predict(texts, batch_size=2)

        assert isinstance(results, list)
        assert len(results) == 5

        for result in results:
            assert isinstance(result, dict)
            assert "sentiment" in result
            assert "confidence" in result
            assert "negative_prob" in result
            assert "positive_prob" in result

            assert result["sentiment"] in ["POSITIVE", "NEGATIVE"]
            assert 0.0 <= result["confidence"] <= 1.0

    def test_batch_predict_different_batch_sizes(self, temp_model_dir):
        """Test batch prediction with different batch sizes."""
        predictor = SentimentPredictor(model_path=str(temp_model_dir), device="cpu")

        texts = ["Text" + str(i) for i in range(10)]

        for batch_size in [1, 2, 5, 10, 20]:
            results = predictor.batch_predict(texts, batch_size=batch_size)
            assert len(results) == 10

    def test_predict_empty_text(self, temp_model_dir):
        """Test prediction on empty text."""
        predictor = SentimentPredictor(model_path=str(temp_model_dir), device="cpu")

        result = predictor.predict("")

        assert isinstance(result, str)
        assert result in ["POSITIVE", "NEGATIVE"]

    def test_predict_long_text(self, temp_model_dir):
        """Test prediction on very long text (tests truncation)."""
        predictor = SentimentPredictor(model_path=str(temp_model_dir), device="cpu")

        long_text = " ".join(["word"] * 500)
        result = predictor.predict(long_text)

        assert isinstance(result, str)
        assert result in ["POSITIVE", "NEGATIVE"]

    def test_predict_special_characters(self, temp_model_dir):
        """Test prediction with special characters."""
        predictor = SentimentPredictor(model_path=str(temp_model_dir), device="cpu")

        texts = [
            "Great product! 😀",
            "Bad quality... @#$%",
            "Mixed\nreviews\twith\nspecial chars"
        ]

        results = predictor.predict(texts)

        assert len(results) == 3
        assert all(r in ["POSITIVE", "NEGATIVE"] for r in results)

    def test_model_in_eval_mode(self, temp_model_dir):
        """Test that model is in eval mode."""
        predictor = SentimentPredictor(model_path=str(temp_model_dir), device="cpu")

        assert not predictor.model.training

    def test_predictions_are_deterministic(self, temp_model_dir):
        """Test that predictions are deterministic."""
        predictor = SentimentPredictor(model_path=str(temp_model_dir), device="cpu")

        text = "This is a test sentence."

        result1 = predictor.predict_with_confidence(text)
        result2 = predictor.predict_with_confidence(text)

        assert result1["sentiment"] == result2["sentiment"]
        assert abs(result1["confidence"] - result2["confidence"]) < 1e-6

    def test_probability_sum_to_one(self, temp_model_dir):
        """Test that probabilities sum to approximately 1."""
        predictor = SentimentPredictor(model_path=str(temp_model_dir), device="cpu")

        result = predictor.predict_with_confidence("Test text")

        prob_sum = result["negative_prob"] + result["positive_prob"]
        assert abs(prob_sum - 1.0) < 0.01

    def test_confidence_matches_max_probability(self, temp_model_dir):
        """Test that confidence equals the maximum probability."""
        predictor = SentimentPredictor(model_path=str(temp_model_dir), device="cpu")

        result = predictor.predict_with_confidence("Test text")

        max_prob = max(result["negative_prob"], result["positive_prob"])
        assert abs(result["confidence"] - max_prob) < 1e-6

    def test_batch_predict_empty_list(self, temp_model_dir):
        """Test batch prediction with empty list."""
        predictor = SentimentPredictor(model_path=str(temp_model_dir), device="cpu")

        results = predictor.batch_predict([])

        assert isinstance(results, list)
        assert len(results) == 0

    def test_device_placement(self, temp_model_dir):
        """Test that tensors are on correct device."""
        predictor = SentimentPredictor(model_path=str(temp_model_dir), device="cpu")

        for param in predictor.model.parameters():
            assert param.device.type == "cpu"
            break

    def test_tokenizer_matches_model(self, temp_model_dir):
        """Test that tokenizer and model are compatible."""
        predictor = SentimentPredictor(model_path=str(temp_model_dir), device="cpu")

        text = "Test compatibility"
        inputs = predictor.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = predictor.model(**inputs)

        assert hasattr(outputs, "logits")
        assert outputs.logits.shape[1] == 2

    @pytest.mark.parametrize("text", [
        "Positive review",
        "Negative review",
        "Neutral statement",
        "123456789",
        "!@#$%^&*()",
    ])
    def test_various_inputs(self, temp_model_dir, text):
        """Test predictor with various input types."""
        predictor = SentimentPredictor(model_path=str(temp_model_dir), device="cpu")

        result = predictor.predict(text)

        assert result in ["POSITIVE", "NEGATIVE"]
