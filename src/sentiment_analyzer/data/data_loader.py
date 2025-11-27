"""Data loading and preparation utilities."""
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from datasets import Dataset


class SentimentDataLoader:
    """Handles loading and splitting of sentiment analysis datasets."""

    def __init__(
        self,
        data_path: str,
        train_split: float = 0.8,
        random_seed: int = 42
    ):
        """
        Initialize data loader.

        Args:
            data_path: Path to the CSV data file
            train_split: Proportion of data for training (default 0.8)
            random_seed: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.train_split = train_split
        self.random_seed = random_seed

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

    def load_and_prepare(
        self,
        sample_size: Optional[int] = None
    ) -> Tuple[Dataset, Dataset, pd.DataFrame]:
        """
        Load CSV and prepare datasets in one call.

        Args:
            sample_size: Optional number of samples to load

        Returns:
            Tuple of (train_dataset, val_dataset, full_dataframe)
        """
        df = self.load_csv(sample_size)
        train_dataset, val_dataset = self.prepare_dataset(df)

        return train_dataset, val_dataset, df
