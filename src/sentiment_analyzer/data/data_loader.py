"""Data loading and preparation utilities."""
import pandas as pd
import warnings
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from datasets import Dataset


class SentimentDataLoader:
    """Handles loading and splitting of sentiment analysis datasets."""

    def __init__(
        self,
        data_path: str,
        train_split: float = 0.6,
        val_split: float = 0.2,
        test_split: float = 0.2,
        random_seed: int = 42
    ):
        """
        Initialize data loader.

        Args:
            data_path: Path to the CSV data file
            train_split: Proportion of data for training (default 0.6)
            val_split: Proportion of data for validation (default 0.2)
            test_split: Proportion of data for testing (default 0.2)
            random_seed: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed

        # Validate splits
        total = train_split + val_split + test_split
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Data splits must sum to 1.0, got {total:.3f} "
                f"(train={train_split}, val={val_split}, test={test_split})"
            )

    def load_csv(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            sample_size: Optional number of samples to load

        Returns:
            DataFrame with loaded data
        """
        df = pd.read_csv(self.data_path)

        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=self.random_seed)

        return df

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        text_column: str = "full_text",
        label_column: str = "label"
    ) -> Tuple[Dataset, Dataset]:
        """
        Prepare train and validation datasets.

        Args:
            df: Input DataFrame
            text_column: Name of the text column
            label_column: Name of the label column

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        train_df, val_df = train_test_split(
            df,
            train_size=self.train_split,
            random_state=self.random_seed,
            stratify=df[label_column]
        )

        train_dataset = Dataset.from_pandas(
            train_df[[text_column, label_column]].reset_index(drop=True)
        )
        val_dataset = Dataset.from_pandas(
            val_df[[text_column, label_column]].reset_index(drop=True)
        )

        return train_dataset, val_dataset

    def prepare_dataset_with_test(
        self,
        df: pd.DataFrame,
        text_column: str = "full_text",
        label_column: str = "label"
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepare train, validation, and test datasets (3-way split).

        Args:
            df: Input DataFrame
            text_column: Name of the text column
            label_column: Name of the label column

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # First split: separate test set from train+val
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_split,
            random_state=self.random_seed,
            stratify=df[label_column]
        )

        # Second split: separate train and val from remaining data
        # Calculate val proportion relative to train+val
        val_ratio = self.val_split / (self.train_split + self.val_split)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=self.random_seed,
            stratify=train_val_df[label_column]
        )

        # Convert to HuggingFace Dataset format
        train_dataset = Dataset.from_pandas(
            train_df[[text_column, label_column]].reset_index(drop=True)
        )
        val_dataset = Dataset.from_pandas(
            val_df[[text_column, label_column]].reset_index(drop=True)
        )
        test_dataset = Dataset.from_pandas(
            test_df[[text_column, label_column]].reset_index(drop=True)
        )

        return train_dataset, val_dataset, test_dataset

    def load_and_prepare(
        self,
        sample_size: Optional[int] = None
    ) -> Tuple[Dataset, Dataset, pd.DataFrame]:
        """
        Load CSV and prepare datasets in one call (2-way split).

        DEPRECATED: Use load_and_prepare_with_test() for 3-way split.

        Args:
            sample_size: Optional number of samples to load

        Returns:
            Tuple of (train_dataset, val_dataset, full_dataframe)
        """
        warnings.warn(
            "load_and_prepare() uses 2-way split and is deprecated. "
            "Use load_and_prepare_with_test() for proper train/val/test split.",
            DeprecationWarning,
            stacklevel=2
        )
        df = self.load_csv(sample_size)
        train_dataset, val_dataset = self.prepare_dataset(df)

        return train_dataset, val_dataset, df

    def load_and_prepare_with_test(
        self,
        sample_size: Optional[int] = None
    ) -> Tuple[Dataset, Dataset, Dataset, pd.DataFrame]:
        """
        Load CSV and prepare datasets with test set (3-way split).

        Args:
            sample_size: Optional number of samples to load

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset, full_dataframe)
        """
        df = self.load_csv(sample_size)
        train_dataset, val_dataset, test_dataset = self.prepare_dataset_with_test(df)

        return train_dataset, val_dataset, test_dataset, df
