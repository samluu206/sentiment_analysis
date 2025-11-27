"""Inference script for sentiment analysis."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentiment_analyzer import SentimentPredictor


def main(args):
    """Main prediction function."""
    model_path = Path(__file__).parent.parent / args.model_path

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    print(f"Loading model from {model_path}")
    predictor = SentimentPredictor(model_path=str(model_path))

    if args.text:
        print("\nInput text:")
        print(f"  {args.text}")
        print("\nPrediction:")

        if args.with_confidence:
            result = predictor.predict_with_confidence(args.text)
            print(f"  Sentiment: {result['sentiment']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Negative probability: {result['negative_prob']:.4f}")
            print(f"  Positive probability: {result['positive_prob']:.4f}")
        else:
            sentiment = predictor.predict(args.text)
            print(f"  Sentiment: {sentiment}")

    elif args.interactive:
        print("\nInteractive mode - Enter text to analyze (type 'quit' to exit):\n")

        while True:
            try:
                text = input("Text: ").strip()

                if text.lower() in ['quit', 'exit', 'q']:
                    print("Exiting...")
                    break

                if not text:
                    continue

                result = predictor.predict_with_confidence(text)
                print(f"  Sentiment: {result['sentiment']}")
                print(f"  Confidence: {result['confidence']:.4f}")
                print()

            except KeyboardInterrupt:
                print("\nExiting...")
                break

    else:
        print("Error: Please provide either --text or --interactive flag")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiment for text")
    parser.add_argument("--model-path", type=str, default="models/final_sentiment_bert",
                        help="Path to trained model")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--with-confidence", action="store_true",
                        help="Show confidence scores")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode for multiple predictions")

    args = parser.parse_args()
    main(args)
