"""Gradio web demo for sentiment analysis."""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import torch

from sentiment_analyzer.inference.predictor import SentimentPredictor


class SentimentDemo:
    """Gradio demo for sentiment analysis."""

    def __init__(self, model_path: str = None):
        """Initialize demo with model."""
        if model_path is None:
            model_path = os.getenv(
                "MODEL_PATH",
                str(Path(__file__).parent.parent.parent.parent / "models" / "roberta_sentiment")
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model from: {model_path}")
        print(f"Using device: {device}")

        self.predictor = SentimentPredictor(model_path=model_path, device=device)

        if device == "cuda":
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPU detected, using CPU")

        print("Model loaded successfully!")

    def predict_sentiment(self, text: str) -> Tuple[str, Dict[str, float], str, str]:
        """
        Predict sentiment for input text.

        Args:
            text: Input review text

        Returns:
            Tuple of (sentiment label, confidence dict, detailed result, character count)
        """
        char_count = f"Character count: {len(text)}/5000"

        if not text or not text.strip():
            return (
                "Error",
                {"NEGATIVE": 0.0, "POSITIVE": 0.0},
                "**Please enter some text to analyze.**\n\nTip: Try one of the example reviews below!",
                char_count
            )

        if len(text) > 5000:
            return (
                "Error",
                {"NEGATIVE": 0.0, "POSITIVE": 0.0},
                f"**Text too long!** Your text has {len(text)} characters, but the maximum is 5,000.\n\nPlease shorten your review.",
                char_count
            )

        result = self.predictor.predict_with_confidence(text)

        sentiment = result["sentiment"]
        confidence = result["confidence"]
        neg_prob = result["negative_prob"]
        pos_prob = result["positive_prob"]

        confidence_dict = {
            "NEGATIVE": neg_prob,
            "POSITIVE": pos_prob
        }

        confidence_bar = "=" * int(confidence * 20)

        confidence_level = ""
        if confidence >= 0.95:
            confidence_level = "Very High"
        elif confidence >= 0.85:
            confidence_level = "High"
        elif confidence >= 0.70:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"

        detailed_result = f"""
## Analysis Result

### Sentiment: **{sentiment}**
**Confidence:** {confidence:.1%} ({confidence_level})

**Confidence Bar:**
`{confidence_bar}` {confidence:.1%}

### Probability Distribution:
- **Negative:** {neg_prob:.2%}
- **Positive:** {pos_prob:.2%}

---
**Review Length:** {len(text)} characters
**Tokens Processed:** ~{len(text.split())} words
        """

        return sentiment, confidence_dict, detailed_result, char_count

    def predict_batch(self, text: str) -> Tuple[str, str]:
        """
        Predict sentiment for multiple texts (one per line).

        Args:
            text: Multiple review texts separated by newlines

        Returns:
            Tuple of (formatted results, statistics summary)
        """
        if not text or not text.strip():
            return (
                "**Please enter texts to analyze** (one per line).\n\nTip: Try the example batch below!",
                ""
            )

        texts = [line.strip() for line in text.split("\n") if line.strip()]

        if not texts:
            return "No valid texts found. Please enter at least one review.", ""

        if len(texts) > 100:
            return (
                f"**Too many texts!** You entered {len(texts)} reviews, but the maximum is 100.\n\nPlease reduce the number of reviews.",
                ""
            )

        results = self.predictor.batch_predict(texts, batch_size=32)

        positive_count = sum(1 for r in results if r["sentiment"] == "POSITIVE")
        negative_count = len(results) - positive_count
        avg_confidence = sum(r["confidence"] for r in results) / len(results)

        stats = f"""
## Batch Analysis Statistics

**Total Reviews:** {len(texts)}
- **Positive:** {positive_count} ({positive_count/len(texts)*100:.1f}%)
- **Negative:** {negative_count} ({negative_count/len(texts)*100:.1f}%)

**Average Confidence:** {avg_confidence:.1%}
        """

        output = f"## Detailed Results ({len(texts)} reviews)\n\n"

        for i, (text, result) in enumerate(zip(texts, results), 1):
            sentiment = result["sentiment"]
            confidence = result["confidence"]
            text_preview = text[:60] + "..." if len(text) > 60 else text

            output += f"### {i}. {text_preview}\n"
            output += f"**Sentiment:** {sentiment} | **Confidence:** {confidence:.1%}\n\n"

        return output, stats

    def clear_single(self):
        """Clear single analysis inputs."""
        return "", None, "", "Character count: 0/5000"

    def clear_batch(self):
        """Clear batch analysis inputs."""
        return "", "", ""

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface."""
        with gr.Blocks(title="Sentiment Analysis Demo") as demo:
            with gr.Row():
                with gr.Column(scale=1, min_width=50):
                    pass

                with gr.Column(scale=8):
                    gr.Markdown(
                        """
                        # Product Review Sentiment Analysis
                        ### Powered by RoBERTa
                        """
                    )

                    with gr.Tabs():
                        # Single prediction tab
                        with gr.Tab("Single Analysis"):
                            gr.Markdown(
                                """
                                ### Analyze a single review
                                Enter your product review below (maximum 5,000 characters).
                                """
                            )

                            single_input = gr.Textbox(
                                label="Review Text",
                                placeholder="Enter your product review here...\n\nExample: This product exceeded my expectations! The quality is outstanding and delivery was fast.",
                                lines=6,
                                max_lines=12
                            )

                            char_counter = gr.Markdown("Character count: 0/5000")

                            single_input.change(
                                fn=lambda text: f"Character count: {len(text)}/5000",
                                inputs=single_input,
                                outputs=char_counter
                            )

                            with gr.Row():
                                single_button = gr.Button("Analyze Sentiment", variant="primary", size="lg")
                                clear_button = gr.Button("Clear", variant="secondary")

                            gr.Markdown("---")
                            gr.Markdown("### Analysis Results")

                            sentiment_output = gr.Label(
                                label="Predicted Sentiment",
                                num_top_classes=2
                            )

                            detailed_output = gr.Markdown(
                                label="Detailed Analysis",
                                value=""
                            )

                            single_button.click(
                                fn=self.predict_sentiment,
                                inputs=single_input,
                                outputs=[sentiment_output, sentiment_output, detailed_output, char_counter]
                            )

                            clear_button.click(
                                fn=self.clear_single,
                                inputs=[],
                                outputs=[single_input, sentiment_output, detailed_output, char_counter]
                            )

                        # Batch prediction tab
                        with gr.Tab("Batch Analysis"):
                            gr.Markdown(
                                """
                                ### Analyze multiple reviews at once
                                Enter multiple reviews, **one per line**. Maximum 100 reviews.

                                **Tip:** Perfect for analyzing survey responses or product feedback!
                                """
                            )

                            batch_input = gr.Textbox(
                                label="Review Texts (one per line)",
                                placeholder="Enter multiple reviews, one per line...\n\nExample:\nGreat product, love it!\nTerrible quality, very disappointed.\nGood value for money.",
                                lines=12,
                                max_lines=25
                            )

                            with gr.Row():
                                batch_button = gr.Button("Analyze All", variant="primary", size="lg")
                                clear_batch_button = gr.Button("Clear", variant="secondary")

                            stats_output = gr.Markdown(label="Statistics Summary")
                            batch_output = gr.Markdown(label="Detailed Results")

                            batch_button.click(
                                fn=self.predict_batch,
                                inputs=batch_input,
                                outputs=[batch_output, stats_output]
                            )

                            clear_batch_button.click(
                                fn=self.clear_batch,
                                inputs=[],
                                outputs=[batch_input, batch_output, stats_output]
                            )

                        # Model info tab
                        with gr.Tab("Model Info"):
                            gr.Markdown(
                                """
                                ## Model Details

                                **Architecture:** RoBERTa (Robustly Optimized BERT Pretraining Approach)
                                - Base model: `roberta-base` (125M parameters)
                                - Fine-tuned on Amazon product reviews
                                - Framework: PyTorch + Hugging Face Transformers

                                ### Performance Metrics (Test Set)
                                - **Accuracy:** 94.53%
                                - **F1 Score:** 0.9452
                                - **ROC-AUC:** 0.9828

                                ### Training Details
                                - **Dataset:** 20,000 Amazon product reviews
                                - **Data Split:** 60% train / 20% validation / 20% test
                                - **Epochs:** 3
                                - **Learning Rate:** 2e-5
                                - **Batch Size:** 16
                                - **Max Sequence Length:** 128 tokens
                                - **Optimizer:** AdamW

                                ### Comparison with Other Models
                                | Model | Accuracy | F1 Score | Model Size | Inference Speed |
                                |-------|----------|----------|------------|----------------|
                                | **RoBERTa** | **94.53%** | **0.945** | **500 MB** | **50ms** |
                                | BERT | 91.88% | 0.922 | 639 MB | 60ms |
                                | DistilBERT | 91.47% | 0.915 | 268 MB | 30ms |
                                """
                            )

                        # Usage guide tab
                        with gr.Tab("Usage Guide"):
                            gr.Markdown(
                                """
                                ### Frequently Asked Questions

                                **Q: What kind of text works best?**
                                A: Product reviews, customer feedback, and opinion text in English.

                                **Q: Can I analyze non-English text?**
                                A: The model is trained on English reviews, so accuracy may vary for other languages.

                                **Q: What if my review is very short?**
                                A: The model works with any length, but 10+ words gives better results.

                                **Q: What does the confidence score mean?**
                                A: It shows how certain the model is. Higher = more confident.

                                **Q: Can I use this for social media posts?**
                                A: Yes! Though it's optimized for product reviews, it works on general sentiment text.

                                **Q: Is my data stored anywhere?**
                                A: No, all processing is done in real-time. Nothing is saved.

                                ### Troubleshooting

                                **Problem:** Results don't seem accurate
                                **Solution:** Try providing more context in your review. Very short or ambiguous text may confuse the model.

                                **Problem:** Text is too long
                                **Solution:** Keep reviews under 5,000 characters. Split longer text into multiple reviews.

                                **Problem:** Batch analysis takes too long
                                **Solution:** Reduce the number of reviews or break into smaller batches of 20-30 reviews.

                                ### Need Help?
                                Check the Model Info tab for technical details or try the example reviews to see how it works!
                                """
                            )

                    gr.Markdown(
                        """
                        ---
                        **Built with:** [Hugging Face Transformers](https://huggingface.co/transformers/) | [Gradio](https://gradio.app/) | [PyTorch](https://pytorch.org/)

                        **Model:** RoBERTa-base fine-tuned on 20K Amazon reviews
                        """
                    )

                with gr.Column(scale=1, min_width=50):
                    pass

        return demo

    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        demo = self.create_interface()
        demo.launch(**kwargs)


def main():
    """Main entry point for demo."""
    demo = SentimentDemo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
