"""Data collection script for Amazon reviews."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentiment_analyzer.data.amazon_api import load_amazon_dataset, collect_diverse_reviews
from sentiment_analyzer.utils.helpers import setup_logging


def main(args):
    """Main data collection function."""
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    project_root = Path(__file__).parent.parent
    output_path = project_root / "data" / "raw" / args.output_file

    logger.info(f"Data collection started")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Output: {output_path}")

    if args.dataset == "amazon_polarity":
        df = load_amazon_dataset(
            dataset_name="amazon_polarity",
            split=args.split,
            sample_size=args.sample_size
        )

    elif args.dataset == "amazon_reviews_multi":
        from datasets import load_dataset

        logger.info("Loading Amazon Reviews Multi dataset")
        dataset = load_dataset(
            "amazon_reviews_multi",
            "en",
            split=args.split
        )

        if args.sample_size and args.sample_size < len(dataset):
            dataset = dataset.select(range(args.sample_size))

        df = dataset.to_pandas()

        if 'review_title' in df.columns and 'review_body' in df.columns:
            df['full_text'] = df['review_title'] + ". " + df['review_body']
            df['full_text'] = df['full_text'].fillna('')

        if 'stars' in df.columns:
            df['label'] = (df['stars'] >= 4).astype(int)
            df['label_name'] = df['label'].map({0: 'Negative', 1: 'Positive'})

        df = df[['full_text', 'label', 'label_name', 'stars', 'product_category']]

    elif args.dataset == "diverse":
        logger.info("Collecting diverse reviews across categories")
        df = collect_diverse_reviews(
            reviews_per_category=args.sample_size // 5 if args.sample_size else 1000
        )

    else:
        logger.error(f"Unknown dataset: {args.dataset}")
        sys.exit(1)

    logger.info(f"Collected {len(df)} reviews")

    logger.info("Data statistics:")
    if 'label' in df.columns:
        logger.info(f"  Label distribution:\n{df['label'].value_counts()}")
    if 'label_name' in df.columns:
        logger.info(f"  Sentiment distribution:\n{df['label_name'].value_counts()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    logger.info("Sample reviews:")
    for i, row in df.head(3).iterrows():
        text = row['full_text'][:100]
        label = row.get('label_name', row.get('label', 'N/A'))
        logger.info(f"  [{label}] {text}...")

    logger.info("Data collection complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect Amazon review data for sentiment analysis"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="amazon_polarity",
        choices=["amazon_polarity", "amazon_reviews_multi", "diverse"],
        help="Dataset to collect"
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of samples to collect"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="amazon_reviews_collected.csv",
        help="Output CSV filename"
    )

    args = parser.parse_args()
    main(args)
