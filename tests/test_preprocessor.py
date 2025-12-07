import pytest
from datasets import Dataset
from src.sentiment_analyzer.data.preprocessor import TextPreprocessor


class TestTextPreprocessor:
    """Tests for TextPreprocessor class."""

    @pytest.fixture
    def model_name(self):
        """Use a small model for testing."""
        return "prajjwal1/bert-tiny"

    @pytest.fixture
    def preprocessor(self, model_name):
        """Create a TextPreprocessor instance."""
        return TextPreprocessor(model_name=model_name, max_length=128)

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        data = {
            "full_text": [
                "This is a great product!",
                "Terrible quality, very disappointed.",
                "Average product, nothing special."
            ],
            "label": [1, 0, 1]
        }
        return Dataset.from_dict(data)

    def test_preprocessor_initialization(self, model_name):
        """Test preprocessor initialization."""
        preprocessor = TextPreprocessor(model_name=model_name, max_length=64)
        assert preprocessor.max_length == 64
        assert preprocessor.tokenizer is not None

    def test_default_max_length(self, model_name):
        """Test default max_length parameter."""
        preprocessor = TextPreprocessor(model_name=model_name)
        assert preprocessor.max_length == 128

    def test_tokenizer_loading(self, preprocessor):
        """Test tokenizer is loaded correctly."""
        tokenizer = preprocessor.get_tokenizer()
        assert tokenizer is not None
        assert hasattr(tokenizer, "encode")
        assert hasattr(tokenizer, "decode")

    def test_tokenize_function(self, preprocessor):
        """Test tokenize_function with sample text."""
        examples = {
            "full_text": ["Hello world!", "How are you?"]
        }

        tokenized = preprocessor.tokenize_function(examples)

        assert "input_ids" in tokenized
        assert "attention_mask" in tokenized
        assert len(tokenized["input_ids"]) == 2
        assert len(tokenized["attention_mask"]) == 2

    def test_tokenize_single_text(self, preprocessor):
        """Test tokenizing a single text."""
        examples = {"full_text": ["This is a test sentence."]}
        tokenized = preprocessor.tokenize_function(examples)

        assert "input_ids" in tokenized
        assert "attention_mask" in tokenized
        assert isinstance(tokenized["input_ids"][0], list)
        assert all(isinstance(id, int) for id in tokenized["input_ids"][0])

    def test_tokenize_respects_max_length(self, preprocessor):
        """Test that tokenization respects max_length."""
        long_text = " ".join(["word"] * 200)
        examples = {"full_text": [long_text]}

        tokenized = preprocessor.tokenize_function(examples)

        assert len(tokenized["input_ids"][0]) <= preprocessor.max_length

    def test_process_dataset(self, preprocessor, sample_dataset):
        """Test processing entire dataset."""
        processed = preprocessor.process_dataset(sample_dataset)

        assert "input_ids" in processed.column_names
        assert "attention_mask" in processed.column_names
        assert "full_text" in processed.column_names
        assert "label" in processed.column_names
        assert len(processed) == len(sample_dataset)

    def test_process_dataset_preserves_labels(self, preprocessor, sample_dataset):
        """Test that labels are preserved after processing."""
        original_labels = sample_dataset["label"]
        processed = preprocessor.process_dataset(sample_dataset)

        assert processed["label"] == original_labels

    def test_tokenization_batched(self, preprocessor):
        """Test batched tokenization."""
        examples = {
            "full_text": [
                "First text.",
                "Second text.",
                "Third text."
            ]
        }

        tokenized = preprocessor.tokenize_function(examples)

        assert len(tokenized["input_ids"]) == 3
        assert len(tokenized["attention_mask"]) == 3

    def test_special_tokens_added(self, preprocessor):
        """Test that special tokens are added."""
        text = "Hello"
        examples = {"full_text": [text]}

        tokenized = preprocessor.tokenize_function(examples)
        tokenizer = preprocessor.get_tokenizer()

        token_ids = tokenized["input_ids"][0]

        if tokenizer.cls_token_id is not None:
            assert tokenizer.cls_token_id in token_ids
        if tokenizer.sep_token_id is not None:
            assert tokenizer.sep_token_id in token_ids

    def test_empty_text_handling(self, preprocessor):
        """Test handling of empty text."""
        examples = {"full_text": [""]}

        tokenized = preprocessor.tokenize_function(examples)

        assert "input_ids" in tokenized
        assert len(tokenized["input_ids"]) == 1
        assert len(tokenized["input_ids"][0]) > 0

    def test_special_characters(self, preprocessor):
        """Test handling of special characters."""
        examples = {
            "full_text": [
                "Text with emoji 😀",
                "Text with symbols @#$%",
                "Text with\nnewlines\nand\ttabs"
            ]
        }

        tokenized = preprocessor.tokenize_function(examples)

        assert "input_ids" in tokenized
        assert len(tokenized["input_ids"]) == 3
        assert all(len(ids) > 0 for ids in tokenized["input_ids"])

    def test_truncation(self, preprocessor):
        """Test that truncation works correctly."""
        very_long_text = " ".join(["word"] * 500)
        examples = {"full_text": [very_long_text]}

        tokenized = preprocessor.tokenize_function(examples)

        assert len(tokenized["input_ids"][0]) <= preprocessor.max_length

    def test_different_max_lengths(self, model_name):
        """Test preprocessor with different max_length values."""
        max_lengths = [32, 64, 128, 256]

        for max_len in max_lengths:
            preprocessor = TextPreprocessor(model_name=model_name, max_length=max_len)
            long_text = " ".join(["word"] * 500)
            examples = {"full_text": [long_text]}

            tokenized = preprocessor.tokenize_function(examples)

            assert len(tokenized["input_ids"][0]) <= max_len

    def test_padding_in_batch(self, preprocessor):
        """Test padding behavior in batch processing."""
        examples = {
            "full_text": [
                "Short.",
                "This is a much longer sentence with many more words."
            ]
        }

        tokenized = preprocessor.tokenize_function(examples)

        ids_lengths = [len(ids) for ids in tokenized["input_ids"]]
        assert ids_lengths[0] != ids_lengths[1]

    def test_process_empty_dataset(self, preprocessor):
        """Test processing an empty dataset (columns not added when empty)."""
        empty_dataset = Dataset.from_dict({"full_text": [], "label": []})

        processed = preprocessor.process_dataset(empty_dataset)

        assert len(processed) == 0
        # Note: Dataset.map() doesn't add columns when dataset is empty
        # This is expected HuggingFace Datasets behavior
        assert "full_text" in processed.column_names
        assert "label" in processed.column_names

    def test_get_tokenizer_returns_same_instance(self, preprocessor):
        """Test that get_tokenizer returns the same instance."""
        tokenizer1 = preprocessor.get_tokenizer()
        tokenizer2 = preprocessor.get_tokenizer()

        assert tokenizer1 is tokenizer2

    @pytest.mark.parametrize("text,expected_tokens_gt", [
        ("hello", 1),
        ("hello world", 2),
        ("this is a longer sentence", 5),
    ])
    def test_tokenization_produces_tokens(self, preprocessor, text, expected_tokens_gt):
        """Test that tokenization produces expected number of tokens."""
        examples = {"full_text": [text]}
        tokenized = preprocessor.tokenize_function(examples)

        num_tokens = len(tokenized["input_ids"][0])

        assert num_tokens >= expected_tokens_gt
