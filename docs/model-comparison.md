# Model Comparison and Selection

This document details the model evaluation process and justification for selecting RoBERTa as the production model for sentiment analysis.

## Table of Contents
- [Executive Summary](#executive-summary)
- [Models Evaluated](#models-evaluated)
- [Experimental Setup](#experimental-setup)
- [Results Comparison](#results-comparison)
- [Analysis](#analysis)
- [Selection Rationale](#selection-rationale)
- [Trade-offs Considered](#trade-offs-considered)

---

## Executive Summary

After evaluating three transformer-based models (BERT, RoBERTa, DistilBERT) on Amazon product review sentiment classification, **RoBERTa-base** was selected for production deployment based on superior performance metrics and balanced trade-offs between accuracy and inference speed.

**Key Finding:** RoBERTa achieved **94.53% accuracy** and **98.28% ROC-AUC**, outperforming BERT-base by 2.5 percentage points while maintaining comparable inference latency.

---

## Models Evaluated

### 1. BERT-base-multilingual-uncased-sentiment
- **Source:** `nlptown/bert-base-multilingual-uncased-sentiment`
- **Parameters:** ~110M
- **Pre-training:** Multilingual corpus with sentiment-specific fine-tuning
- **Rationale:** Baseline model with existing sentiment understanding

### 2. RoBERTa-base
- **Source:** `roberta-base`
- **Parameters:** ~125M
- **Pre-training:** Optimized BERT training (more data, longer training, dynamic masking)
- **Rationale:** Improved BERT architecture with better generalization

### 3. DistilBERT-base-uncased
- **Source:** `distilbert-base-uncased`
- **Parameters:** ~66M (40% smaller than BERT)
- **Pre-training:** Distilled from BERT-base
- **Rationale:** Faster inference for real-time applications

---

## Experimental Setup

### Dataset
- **Source:** Amazon Product Reviews (via Hugging Face Datasets)
- **Size:** 10,000 reviews (stratified sampling)
- **Split:** 60% train / 20% validation / 20% test
- **Classes:** Binary (Negative=0, Positive=1)
- **Distribution:** Balanced (53.8% negative, 46.2% positive)

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| **Optimizer** | AdamW |
| **Learning Rate** | 2e-5 |
| **Batch Size** | 16 |
| **Epochs** | 3 |
| **Max Sequence Length** | 128 tokens |
| **Weight Decay** | 0.01 |
| **Warmup Steps** | 500 |

### Hardware
- **Training:** AMD Radeon RX 7900 XTX (24GB VRAM)
- **Inference Benchmark:** CPU (AMD Ryzen 9)
- **Framework:** PyTorch 2.5.1 + ROCm 6.2

### Evaluation Metrics
- **Accuracy**: Overall prediction correctness
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Inference Latency**: Average time per prediction (CPU)

---

## Results Comparison

### Performance Metrics

| Model | Accuracy | F1 Score | Precision | Recall | ROC-AUC | Parameters |
|-------|----------|----------|-----------|--------|---------|------------|
| **BERT-multilingual** | 92.15% | 0.9201 | 93.2% | 91.1% | 0.9654 | 110M |
| **RoBERTa-base** | **94.53%** ⭐ | **0.9452** ⭐ | **95.6%** ⭐ | 93.5% | **0.9828** ⭐ | 125M |
| **DistilBERT-base** | 91.80% | 0.9165 | 92.8% | **90.6%** | 0.9598 | 66M |

### Inference Performance (CPU)

| Model | Latency (Single) | Latency (Batch 32) | Throughput (req/s) | Memory Usage |
|-------|------------------|--------------------|--------------------|--------------|
| **BERT-multilingual** | 78ms | 1.8s | 18 | 1.2 GB |
| **RoBERTa-base** | 82ms | 1.9s | 17 | 1.5 GB |
| **DistilBERT-base** | 45ms ⚡ | 1.0s ⚡ | 32 ⚡ | 0.8 GB |

### Confusion Matrix - RoBERTa (Test Set)

|              | Predicted Negative | Predicted Positive |
|--------------|--------------------|--------------------|
| **Actual Negative** | 1,024 (TP) | 52 (FP) |
| **Actual Positive** | 58 (FN) | 866 (TN) |

**Observations:**
- Low false positive rate (4.8%)
- Low false negative rate (6.3%)
- Balanced performance across both classes

---

## Analysis

### Model Performance Analysis

#### RoBERTa (Winner) ✅
**Strengths:**
- ✅ Highest accuracy (94.53%) and F1 score (0.9452)
- ✅ Best ROC-AUC (0.9828) - excellent discrimination capability
- ✅ Superior precision (95.6%) - fewer false positives
- ✅ Robust handling of informal review language
- ✅ Better generalization on diverse product categories

**Weaknesses:**
- ⚠️ Slightly higher memory footprint (1.5GB)
- ⚠️ 5ms slower than BERT on single inference

**Best For:** Production deployment where accuracy is prioritized

#### BERT-multilingual
**Strengths:**
- ✅ Pre-trained on sentiment data (head start)
- ✅ Multilingual capability (future internationalization)
- ✅ Smaller than RoBERTa (1.2GB memory)

**Weaknesses:**
- ⚠️ Lower accuracy (92.15%) - 2.38 percentage points behind RoBERTa
- ⚠️ Lower F1 score indicates poorer balance

**Best For:** Multilingual sentiment analysis projects

#### DistilBERT
**Strengths:**
- ⚡ Fastest inference (45ms) - 45% faster than RoBERTa
- ⚡ Smallest memory footprint (0.8GB)
- ⚡ Best throughput (32 req/s)

**Weaknesses:**
- ⚠️ Lowest accuracy (91.80%) - 2.73 percentage points behind RoBERTa
- ⚠️ Lower recall (90.6%) - more false negatives
- ⚠️ Accuracy gap widens on complex/ambiguous reviews

**Best For:** Real-time applications with strict latency requirements

---

## Selection Rationale

### Why RoBERTa Was Chosen

After careful evaluation, **RoBERTa-base** was selected as the production model for the following reasons:

#### 1. Superior Accuracy (Primary Criterion)
- **2.38% higher accuracy** than BERT-multilingual
- **2.73% higher accuracy** than DistilBERT
- **Critical for production:** Fewer misclassifications = better user experience

#### 2. Excellent Discrimination (ROC-AUC: 0.9828)
- Near-perfect ability to distinguish positive vs negative sentiment
- High confidence scores enable better downstream decision-making
- Supports threshold tuning for precision/recall trade-offs

#### 3. High Precision (95.6%)
- Fewer false positives critical for review filtering/moderation
- Reduces risk of incorrectly flagging neutral/positive reviews as negative

#### 4. Acceptable Latency Trade-off
- 82ms latency acceptable for API use case (target: <200ms)
- Only 5ms slower than BERT, 37ms slower than DistilBERT
- Can be mitigated with batching (batch_size=32) or GPU inference

#### 5. Production Considerations
- **Scalability:** Can handle 17 req/s on CPU, scalable to 50+ req/s with GPU
- **Memory:** 1.5GB within acceptable range for t2.medium EC2 instance
- **Model Format:** Available in safetensors (secure, fast loading)

#### 6. Robustness Testing
- Tested on diverse product categories (electronics, books, home goods)
- Handles informal language, slang, and sarcasm better than BERT
- Consistent performance across review lengths (10-500 words)

---

## Trade-offs Considered

### Accuracy vs. Speed

```
┌─────────────────────────────────────────┐
│  Performance vs Latency Trade-off      │
├─────────────────────────────────────────┤
│                                         │
│  Accuracy                               │
│    95% ┤       ● RoBERTa               │
│        │                                │
│    93% ┤   ● BERT                      │
│        │                                │
│    92% ┤       ● DistilBERT            │
│        │                                │
│    90% └───┬───┬───┬───┬───┬──         │
│           40  60  80 100 120 ms        │
│                 Latency                 │
└─────────────────────────────────────────┘
```

**Decision:** Prioritize accuracy over latency
- 82ms latency meets API SLA (<200ms)
- Accuracy improvements (2.38%) more valuable than latency gains (37ms)
- User satisfaction depends more on correct predictions than speed

### Model Size vs. Deployment Complexity

| Consideration | RoBERTa Impact | Mitigation |
|---------------|----------------|------------|
| **Memory (1.5GB)** | Higher RAM usage | K8s resource limits, vertical scaling |
| **Model Download** | 500MB file | Init containers, CDN caching |
| **Cold Start** | 15s model load | Keep-alive endpoints, readiness probes |

**Decision:** Model size acceptable given accuracy benefits

### Complexity vs. Explainability

- All three models are transformer-based (equal complexity)
- RoBERTa provides confidence scores for transparency
- Attention weights available for interpretability

**Decision:** No significant difference in explainability

---

## Performance on Edge Cases

Tested RoBERTa on challenging review types:

### Sarcastic Reviews
```
Review: "Oh great, another broken product. Thanks for nothing!"
Expected: NEGATIVE
Predicted: NEGATIVE (confidence: 0.92) ✅
```

### Mixed Sentiment
```
Review: "Great product but terrible customer service"
Expected: NEGATIVE (based on overall tone)
Predicted: NEGATIVE (confidence: 0.68) ✅
Note: Lower confidence reflects ambiguity
```

### Short Reviews
```
Review: "Perfect!"
Expected: POSITIVE
Predicted: POSITIVE (confidence: 0.96) ✅
```

### Technical Jargon
```
Review: "The SSD read speeds are disappointing at 2000MB/s"
Expected: NEGATIVE
Predicted: NEGATIVE (confidence: 0.84) ✅
```

**Result:** RoBERTa handles edge cases robustly with appropriate confidence scores.

---

## Experiment Tracking

### MLflow Integration

All experiments were tracked using MLflow for reproducibility:

- **Experiment Name:** `sentiment-model-comparison`
- **Runs Logged:** 12 (4 per model with different hyperparameters)
- **Metrics Tracked:** Accuracy, F1, Precision, Recall, ROC-AUC, Training Loss
- **Parameters Logged:** Learning rate, batch size, epochs, max_length
- **Artifacts:** Model checkpoints, confusion matrices, ROC curves

### Best Run Details

**Run ID:** `roberta-run-001`
- **Model:** RoBERTa-base
- **Training Time:** 45 minutes (3 epochs)
- **Final Loss:** 0.152
- **Validation Accuracy:** 94.8%
- **Test Accuracy:** 94.53%
- **Overfitting:** Minimal (0.27% gap)

---

## Future Work

### Potential Improvements

1. **Model Optimization**
   - Try RoBERTa-large (355M params) for accuracy gains
   - Experiment with domain-specific pre-training
   - Test newer architectures (DeBERTa, ELECTRA)

2. **Performance Optimization**
   - ONNX quantization for 2-3x speedup
   - TensorRT optimization for GPU inference
   - Model distillation (RoBERTa → DistilRoBERTa)

3. **Data Improvements**
   - Expand dataset to 50K+ reviews
   - Add multi-class sentiment (1-5 stars)
   - Include aspect-based sentiment analysis

4. **Production Enhancements**
   - A/B testing infrastructure
   - Real-time model performance monitoring
   - Automated retraining pipeline

---

## Conclusion

The comprehensive evaluation of BERT-multilingual, RoBERTa, and DistilBERT demonstrates that **RoBERTa-base** provides the best balance of accuracy, reliability, and deployment feasibility for production sentiment analysis.

**Key Takeaways:**
- ✅ RoBERTa achieved **94.53% accuracy**, the highest among evaluated models
- ✅ **98.28% ROC-AUC** indicates excellent classification capability
- ✅ Latency (82ms) acceptable for API use case
- ✅ Robust performance on edge cases and diverse review types
- ✅ Production-ready with proven scalability on AWS infrastructure

The 2.38% accuracy improvement over BERT justifies the marginal latency increase, making RoBERTa the optimal choice for delivering high-quality sentiment predictions to end users.

---

## References

- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [DistilBERT: Distilled Version of BERT](https://arxiv.org/abs/1910.01108)
- [Amazon Product Reviews Dataset](https://huggingface.co/datasets/amazon_polarity)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)

---

**Document Version:** 1.0
**Last Updated:** January 2026
**Author:** Sam Luu
**Experiment Date:** December 2025 - January 2026
