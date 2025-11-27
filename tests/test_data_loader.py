"""Unit tests for data loading module."""
import pandas as pd
import pytest
from pathlib import Path

from sentiment_analyzer.data.data_loader import SentimentDataLoader


class TestSentimentDataLoader:
    """Test cases for SentimentDataLoader class."""

    @pytest.fixture
    def sample_csv_file(self, tmp_path):
        """Create a temporary CSV file for testing."""
        df = pd.DataFrame({
            'full_text': [
                'Great product! Very satisfied.',
                'Poor quality, not recommended.',
                'Good value for money.',
                'Excellent service!',
                'Disappointed with purchase.',
                'Amazing quality!'
            ],
            'label': [1, 0, 1, 1, 0, 1],
            'label_name': ['Positive', 'Negative', 'Positive', 'Positive', 'Negative', 'Positive']
        })

        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)

        return csv_path

    @pytest.fixture
    def data_loader(self, sample_csv_file):
        """Create SentimentDataLoader instance."""
        return SentimentDataLoader(
            data_path=str(sample_csv_file),
            train_split=0.8,
            random_seed=42
        )

    def test_load_csv(self, data_loader):
        """Test CSV loading."""
        df = data_loader.load_csv()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6
        assert 'full_text' in df.columns
        assert 'label' in df.columns

    def test_load_csv_with_sample_size(self, data_loader):
        """Test CSV loading with sample size limit."""
        df = data_loader.load_csv(sample_size=3)

        assert len(df) == 3

    def test_prepare_dataset(self, data_loader):
        """Test dataset preparation and splitting."""
        df = data_loader.load_csv()
        train_dataset, val_dataset = data_loader.prepare_dataset(df)

        assert len(train_dataset) == 4
        assert len(val_dataset) == 2

        assert 'full_text' in train_dataset.column_names
        assert 'label' in train_dataset.column_names

    def test_load_and_prepare(self, data_loader):
        """Test combined loading and preparation."""
        train_dataset, val_dataset, df = data_loader.load_and_prepare()

        assert isinstance(df, pd.DataFrame)
        assert len(train_dataset) > 0
        assert len(val_dataset) > 0
        assert len(df) == len(train_dataset) + len(val_dataset)

    def test_stratified_split(self, data_loader):
        """Test that split maintains class distribution."""
        df = data_loader.load_csv()
        train_dataset, val_dataset = data_loader.prepare_dataset(df)

        original_positive_ratio = df['label'].mean()

        train_df = train_dataset.to_pandas()
        train_positive_ratio = train_df['label'].mean()

        assert abs(original_positive_ratio - train_positive_ratio) < 0.3
