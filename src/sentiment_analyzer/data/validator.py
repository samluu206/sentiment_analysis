"""Data validation utilities."""
import logging
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates sentiment analysis datasets."""

    def __init__(
        self,
        text_column: str = "full_text",
        label_column: str = "label",
        min_text_length: int = 10,
        max_text_length: int = 5000
    ):
        """
        Initialize validator.

        Args:
            text_column: Name of text column
            label_column: Name of label column
            min_text_length: Minimum allowed text length
            max_text_length: Maximum allowed text length
        """
        self.text_column = text_column
        self.label_column = label_column
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame schema.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []

        if self.text_column not in df.columns:
            errors.append(f"Missing required column: {self.text_column}")

        if self.label_column not in df.columns:
            errors.append(f"Missing required column: {self.label_column}")

        return len(errors) == 0, errors

    def validate_text_quality(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate and clean text data.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (cleaned DataFrame, validation statistics)
        """
        stats = {
            'original_count': len(df),
            'null_text': 0,
            'empty_text': 0,
            'too_short': 0,
            'too_long': 0,
            'removed_count': 0
        }

        df = df.copy()

        stats['null_text'] = df[self.text_column].isnull().sum()
        df = df.dropna(subset=[self.text_column])

        df[self.text_column] = df[self.text_column].str.strip()

        stats['empty_text'] = (df[self.text_column] == '').sum()
        df = df[df[self.text_column] != '']

        text_lengths = df[self.text_column].str.len()

        stats['too_short'] = (text_lengths < self.min_text_length).sum()
        stats['too_long'] = (text_lengths > self.max_text_length).sum()

        df = df[
            (text_lengths >= self.min_text_length) &
            (text_lengths <= self.max_text_length)
        ]

        stats['removed_count'] = stats['original_count'] - len(df)
        stats['final_count'] = len(df)

        return df, stats

    def validate_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate label distribution.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (DataFrame, label statistics)
        """
        stats = {
            'null_labels': 0,
            'invalid_labels': 0,
            'class_distribution': {},
            'is_balanced': False
        }

        df = df.copy()

        stats['null_labels'] = df[self.label_column].isnull().sum()
        df = df.dropna(subset=[self.label_column])

        valid_labels = {0, 1}
        invalid_mask = ~df[self.label_column].isin(valid_labels)
        stats['invalid_labels'] = invalid_mask.sum()

        df = df[~invalid_mask]

        label_counts = df[self.label_column].value_counts()
        stats['class_distribution'] = label_counts.to_dict()

        if len(label_counts) == 2:
            ratio = label_counts.min() / label_counts.max()
            stats['imbalance_ratio'] = ratio
            stats['is_balanced'] = ratio >= 0.7

        return df, stats

    def validate_dataset(
        self,
        df: pd.DataFrame,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform complete dataset validation.

        Args:
            df: Input DataFrame
            verbose: Whether to log validation results

        Returns:
            Tuple of (validated DataFrame, validation report)
        """
        if verbose:
            logger.info("Starting dataset validation")

        is_valid, errors = self.validate_schema(df)
        if not is_valid:
            error_msg = "Schema validation failed: " + "; ".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        df, text_stats = self.validate_text_quality(df)
        if verbose:
            logger.info(f"Text validation: {text_stats}")

        df, label_stats = self.validate_labels(df)
        if verbose:
            logger.info(f"Label validation: {label_stats}")

        report = {
            'text_validation': text_stats,
            'label_validation': label_stats,
            'final_dataset_size': len(df)
        }

        if verbose:
            logger.info(f"Validation complete. Final dataset size: {len(df)}")
            logger.info(f"Class distribution: {label_stats['class_distribution']}")
            if not label_stats['is_balanced']:
                logger.warning(
                    f"Dataset is imbalanced (ratio: {label_stats.get('imbalance_ratio', 0):.2f})"
                )

        return df, report

    def detect_duplicates(
        self,
        df: pd.DataFrame,
        remove: bool = True
    ) -> Tuple[pd.DataFrame, int]:
        """
        Detect and optionally remove duplicate texts.

        Args:
            df: Input DataFrame
            remove: Whether to remove duplicates

        Returns:
            Tuple of (DataFrame, number of duplicates found)
        """
        duplicate_count = df[self.text_column].duplicated().sum()

        if remove and duplicate_count > 0:
            logger.info(f"Removing {duplicate_count} duplicate texts")
            df = df.drop_duplicates(subset=[self.text_column], keep='first')

        return df, duplicate_count
