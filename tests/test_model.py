import pytest
import torch
from pathlib import Path
import tempfile
import shutil
from src.sentiment_analyzer.models.model import SentimentModel


class TestSentimentModel:
    """Tests for SentimentModel class."""

    @pytest.fixture
    def model_name(self):
        """Use a small model for testing."""
        return "prajjwal1/bert-tiny"

    @pytest.fixture
    def sentiment_model(self, model_name):
        """Create a SentimentModel instance."""
        return SentimentModel(
            model_name=model_name,
            num_labels=2,
            device="cpu"
        )

    def test_model_initialization(self, model_name):
        """Test model initialization with different parameters."""
        model = SentimentModel(model_name=model_name, num_labels=2)
        assert model.model_name == model_name
        assert model.num_labels == 2
        assert model.model is None

    def test_device_detection_cpu(self, model_name):
        """Test device detection defaults to CPU when specified."""
        model = SentimentModel(model_name=model_name, device="cpu")
        assert model.device == "cpu"

    def test_load_model(self, sentiment_model):
        """Test model loading."""
        model = sentiment_model.load_model()
        assert model is not None
        assert sentiment_model.model is not None
        assert hasattr(model, "forward")

    def test_get_model_auto_loads(self, sentiment_model):
        """Test get_model automatically loads if not loaded."""
        assert sentiment_model.model is None
        model = sentiment_model.get_model()
        assert model is not None
        assert sentiment_model.model is not None

    def test_model_output_shape(self, sentiment_model):
        """Test model output has correct shape."""
        model = sentiment_model.load_model()
        batch_size = 2
        seq_length = 10

        dummy_input = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
            "attention_mask": torch.ones(batch_size, seq_length)
        }

        with torch.no_grad():
            outputs = model(**dummy_input)

        assert hasattr(outputs, "logits")
        assert outputs.logits.shape == (batch_size, 2)

    def test_get_num_parameters(self, sentiment_model):
        """Test parameter counting."""
        num_params = sentiment_model.get_num_parameters()
        assert isinstance(num_params, int)
        assert num_params > 0

    def test_get_trainable_parameters(self, sentiment_model):
        """Test trainable parameter counting."""
        trainable_params = sentiment_model.get_trainable_parameters()
        total_params = sentiment_model.get_num_parameters()

        assert isinstance(trainable_params, int)
        assert trainable_params > 0
        assert trainable_params <= total_params

    def test_save_model(self, sentiment_model):
        """Test model saving."""
        sentiment_model.load_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            sentiment_model.save_model(str(save_path))

            assert save_path.exists()
            assert (save_path / "config.json").exists()
            assert any(save_path.glob("*.bin")) or any(save_path.glob("*.safetensors"))

    def test_save_model_without_loading(self, sentiment_model):
        """Test saving without loading raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No model loaded to save"):
                sentiment_model.save_model(tmpdir)

    def test_load_saved_model(self, sentiment_model):
        """Test loading a saved model."""
        sentiment_model.load_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            sentiment_model.save_model(str(save_path))

            new_model = SentimentModel(
                model_name=str(save_path),
                num_labels=2,
                device="cpu"
            )
            loaded_model = new_model.load_model()

            assert loaded_model is not None
            assert hasattr(loaded_model, "forward")

    def test_model_device_placement(self, sentiment_model):
        """Test model is placed on correct device."""
        model = sentiment_model.load_model()

        for param in model.parameters():
            assert param.device.type == "cpu"
            break

    @pytest.mark.parametrize("num_labels", [2, 3, 5])
    def test_different_num_labels(self, model_name, num_labels):
        """Test model with different number of labels."""
        model = SentimentModel(
            model_name=model_name,
            num_labels=num_labels,
            device="cpu"
        )
        loaded_model = model.load_model()

        dummy_input = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10)
        }

        with torch.no_grad():
            outputs = loaded_model(**dummy_input)

        assert outputs.logits.shape == (1, num_labels)

    def test_model_forward_pass(self, sentiment_model):
        """Test complete forward pass."""
        model = sentiment_model.load_model()
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 15))
        attention_mask = torch.ones(2, 15)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        assert outputs.logits.shape == (2, 2)
        assert not torch.isnan(outputs.logits).any()
        assert not torch.isinf(outputs.logits).any()
