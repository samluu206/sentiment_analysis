"""Unit tests for data validation module."""
import pandas as pd
import pytest

from sentiment_analyzer.data.validator import DataValidator


class TestDataValidator:
    """Test cases for DataValidator class."""

    @pytest.fixture
    def sample_valid_data(self):
        """Create sample valid data."""
        return pd.DataFrame({
            'full_text': [
                'This is a great product! Highly recommend.',
                'Terrible quality. Waste of money.',
                'Decent product for the price.',
                'Amazing service and fast delivery!'
            ],
            'label': [1, 0, 1, 1],
            'label_name': ['Positive', 'Negative', 'Positive', 'Positive']
        })

    @pytest.fixture
    def sample_invalid_data(self):
        """Create sample data with issues."""
        return pd.DataFrame({
            'full_text': [
                'Good product',
                '',
                None,
                'Short',
                'A' * 6000,
                'Another good product'
            ],
            'label': [1, 0, 1, None, 2, 1]
        })

    @pytest.fixture
    def validator(self):
        """Create DataValidator instance."""
        return DataValidator(
            text_column='full_text',
            label_column='label',
            min_text_length=10,
            max_text_length=5000
        )

    def test_validate_schema_valid(self, validator, sample_valid_data):
        """Test schema validation with valid data."""
        is_valid, errors = validator.validate_schema(sample_valid_data)

        assert is_valid
        assert len(errors) == 0

    def test_validate_schema_missing_columns(self, validator):
        """Test schema validation with missing columns."""
        df = pd.DataFrame({'other_column': [1, 2, 3]})

        is_valid, errors = validator.validate_schema(df)

        assert not is_valid
        assert len(errors) == 2
        assert 'full_text' in str(errors)
        assert 'label' in str(errors)

    def test_validate_text_quality(self, validator, sample_invalid_data):
        """Test text quality validation."""
        df_clean, stats = validator.validate_text_quality(sample_invalid_data)

        assert stats['null_text'] == 1
        assert stats['empty_text'] > 0 or stats['too_short'] > 0
        assert stats['too_long'] == 1
        assert len(df_clean) < len(sample_invalid_data)

    def test_validate_labels(self, validator, sample_valid_data):
        """Test label validation."""
        df_clean, stats = validator.validate_labels(sample_valid_data)

        assert stats['null_labels'] == 0
        assert stats['invalid_labels'] == 0
        assert len(stats['class_distribution']) == 2

    def test_validate_labels_with_invalid(self, validator, sample_invalid_data):
        """Test label validation with invalid labels."""
        df_clean, stats = validator.validate_labels(sample_invalid_data)

        assert stats['null_labels'] == 1
        assert stats['invalid_labels'] == 1
        assert len(df_clean) < len(sample_invalid_data)

    def test_detect_duplicates(self, validator):
        """Test duplicate detection."""
        df = pd.DataFrame({
            'full_text': [
                'This is a great product',
                'This is a great product',
                'Different review text'
            ],
            'label': [1, 1, 0]
        })

        df_clean, dup_count = validator.detect_duplicates(df, remove=True)

        assert dup_count == 1
        assert len(df_clean) == 2

    def test_validate_dataset_complete(self, validator, sample_valid_data):
        """Test complete dataset validation."""
        df_clean, report = validator.validate_dataset(sample_valid_data, verbose=False)

        assert 'text_validation' in report
        assert 'label_validation' in report
        assert 'final_dataset_size' in report
        assert report['final_dataset_size'] > 0
