"""Text preprocessing and tokenization utilities."""
from transformers import AutoTokenizer
from datasets import Dataset


class TextPreprocessor:
    """Handles text preprocessing and tokenization."""

    def __init__(
        self,
        model_name: str,
        max_length: int = 128
    ):
        """
        Initialize preprocessor with tokenizer.

        Args:
            model_name: Name of the pretrained model for tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def tokenize_function(self, examples: dict) -> dict:
        """
        Tokenize text examples.

        Args:
            examples: Dictionary with 'full_text' key

        Returns:
            Tokenized examples
        """
        return self.tokenizer(
            examples["full_text"],
            truncation=True,
            max_length=self.max_length
        )

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """
        Apply tokenization to entire dataset.

        Args:
            dataset: Input dataset

        Returns:
            Tokenized dataset
        """
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True
        )
        return tokenized_dataset

    def get_tokenizer(self):
        """Return the tokenizer instance."""
        return self.tokenizer
