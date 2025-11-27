# Data Collection Guide

This guide explains how to collect Amazon product reviews for training the sentiment analysis model.

## Table of Contents

- [Overview](#overview)
- [Data Sources](#data-sources)
- [Collection Methods](#collection-methods)
- [Usage Examples](#usage-examples)
- [Data Validation](#data-validation)

## Overview

The sentiment analyzer requires labeled product reviews for training. This guide covers multiple approaches to collecting Amazon review data, from public datasets to API integration.

## Data Sources

### 1. Public Datasets (Recommended)

The easiest and most reliable approach is using public Amazon review datasets from Hugging Face.

Available datasets:
- **amazon_polarity**: 4M reviews, binary sentiment (positive/negative)
- **amazon_reviews_multi**: Multilingual reviews with ratings
- **amazon_us_reviews**: Category-specific US reviews

Advantages:
- No API keys required
- Pre-labeled data
- Large scale (millions of reviews)
- Legal and compliant

### 2. Amazon Product Advertising API

Amazon's official API for product data.

Limitations:
- Requires approved Amazon Associates account
- Does NOT provide review text directly
- Only provides product metadata and ratings
- Rate limited

Status: Limited usefulness for this project

### 3. Web Scraping

Scraping Amazon product pages directly.

Warnings:
- May violate Amazon Terms of Service
- Subject to blocking/rate limiting
- Requires maintenance as site changes
- Legal gray area

Status: Not recommended for production

## Collection Methods

### Method 1: Using Public Datasets (Recommended)

#### Amazon Polarity Dataset

```bash
python scripts/collect_data.py \
    --dataset amazon_polarity \
    --sample-size 10000 \
    --output-file amazon_reviews_10k.csv
```

This will:
- Download 10,000 reviews from Hugging Face
- Save to `data/raw/amazon_reviews_10k.csv`
- Include binary labels (0=negative, 1=positive)

#### Amazon Reviews Multi (with categories)

```bash
python scripts/collect_data.py \
    --dataset amazon_reviews_multi \
    --sample-size 20000 \
    --output-file amazon_reviews_multi.csv
```

Features:
- Includes product categories
- Star ratings (1-5)
- English reviews
- Review title and body

### Method 2: Category-Specific Collection

To collect reviews from specific product categories:

```python
from sentiment_analyzer.data.amazon_api import collect_diverse_reviews

df = collect_diverse_reviews(
    categories=["electronics", "books", "home"],
    reviews_per_category=5000
)

df.to_csv("data/raw/diverse_reviews.csv", index=False)
```

### Method 3: Custom Dataset Loading

To use your own dataset:

```python
from sentiment_analyzer import SentimentDataLoader

loader = SentimentDataLoader(
    data_path="data/raw/my_reviews.csv",
    train_split=0.8,
    random_seed=42
)

train_ds, val_ds, df = loader.load_and_prepare()
```

Required columns:
- `full_text`: Review text
- `label`: Binary label (0 or 1)

## Usage Examples

### Example 1: Quick Start with 1K Samples

```bash
python scripts/collect_data.py \
    --dataset amazon_polarity \
    --sample-size 1000 \
    --output-file quick_test.csv
```

### Example 2: Large-Scale Collection

```bash
python scripts/collect_data.py \
    --dataset amazon_polarity \
    --sample-size 50000 \
    --split train \
    --output-file amazon_50k_train.csv
```

### Example 3: Programmatic Collection

```python
from sentiment_analyzer.data.amazon_api import load_amazon_dataset
from sentiment_analyzer.data.validator import DataValidator

df = load_amazon_dataset(
    dataset_name="amazon_polarity",
    split="train",
    sample_size=10000
)

validator = DataValidator()
df_clean, report = validator.validate_dataset(df)

print(f"Collected {len(df_clean)} valid reviews")
print(f"Class distribution: {report['label_validation']['class_distribution']}")

df_clean.to_csv("data/raw/validated_reviews.csv", index=False)
```

## Data Validation

### Automatic Validation

The collection scripts automatically validate:
- Text length (10-5000 characters)
- Label validity (0 or 1)
- Missing values
- Duplicates

### Manual Validation

```python
from sentiment_analyzer.data.validator import DataValidator
import pandas as pd

df = pd.read_csv("data/raw/amazon_reviews.csv")

validator = DataValidator(
    text_column="full_text",
    label_column="label",
    min_text_length=10,
    max_text_length=5000
)

df_clean, report = validator.validate_dataset(df, verbose=True)

print("Validation Report:")
print(f"  Original: {report['text_validation']['original_count']} reviews")
print(f"  Removed: {report['text_validation']['removed_count']} reviews")
print(f"  Final: {report['final_dataset_size']} reviews")
print(f"  Class balance ratio: {report['label_validation'].get('imbalance_ratio', 'N/A')}")

df_clean.to_csv("data/processed/cleaned_reviews.csv", index=False)
```

## Amazon Product Advertising API Setup (Optional)

If you want to experiment with the Amazon API for product metadata:

### 1. Sign Up for Amazon Associates

1. Go to [Amazon Associates](https://affiliate-program.amazon.com/)
2. Create an account
3. Complete the application process

### 2. Get Product Advertising API Credentials

1. Visit [Product Advertising API](https://webservices.amazon.com/paapi5/documentation/)
2. Request API access
3. Obtain:
   - Access Key ID
   - Secret Access Key
   - Associate Tag

### 3. Configure Credentials

Create `.env` file:

```bash
AMAZON_ACCESS_KEY=your_access_key_here
AMAZON_SECRET_KEY=your_secret_key_here
AMAZON_PARTNER_TAG=your_associate_tag_here
```

Note: The API does not provide review text, only product metadata.

## Best Practices

1. Start Small: Test with 1K-10K samples before scaling up
2. Validate Data: Always run validation after collection
3. Monitor Balance: Check class distribution (aim for 40-60% balance)
4. Version Data: Track dataset versions with timestamps
5. Document Sources: Record data source and collection date

## Troubleshooting

### Issue: Dataset download fails

Solution: Check internet connection and Hugging Face availability

```python
from datasets import load_dataset

load_dataset("amazon_polarity", split="train[:100]")
```

### Issue: Out of memory

Solution: Use smaller sample sizes or process in batches

```bash
python scripts/collect_data.py --sample-size 1000
```

### Issue: Imbalanced classes

Solution: Use stratified sampling or rebalancing techniques

```python
from sklearn.utils import resample

negative = df[df['label'] == 0]
positive = df[df['label'] == 1]

negative_downsampled = resample(
    negative,
    n_samples=len(positive),
    random_state=42
)

df_balanced = pd.concat([negative_downsampled, positive])
```

## Next Steps

After collecting data:

1. Review the data quality
2. Run the training script
3. Evaluate model performance
4. Iterate with more data if needed

See [TRAINING.md](TRAINING.md) for the next steps.
