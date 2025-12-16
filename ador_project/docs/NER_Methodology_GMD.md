# Global Methodology Document (GMD)
## Fine-Tuning NER Models for Financial Entity Extraction

**Document Type:** Global Methodology Document (GMD)  
**Author:** Ali Dakheel
**Organization:** CMI Architecture & Innovation Team  
**Version:** 1.0

---

## 1. Executive Summary

This document outlines a comprehensive methodology for fine-tuning Named Entity Recognition (NER) models to extract financial entities from semi-structured chat and messaging data. The approach balances accuracy, performance, and production viability while providing a clear path from prototype to deployment.

**Key Objectives:**
- Extract financial entities (ISIN, counterparty, notional, etc.) with >95% F1 score
- Process chat messages in <100ms for real-time applications
- Maintain model performance across diverse message formats
- Enable continuous improvement through active learning

---

## 2. Model Selection Rationale

### 2.1 Candidate Models Analysis

| Model | Strengths | Weaknesses | Suitability |
|-------|-----------|------------|-------------|
| **spaCy (en_core_web_sm)** | Fast inference, lightweight, easy deployment | Limited financial domain knowledge | ⭐⭐⭐ Good baseline |
| **FinBERT-NER** | Pre-trained on financial text | Heavier model, requires GPU | ⭐⭐⭐⭐ Recommended |
| **BERT-base-cased** | Strong general NER, widely supported | Requires significant fine-tuning | ⭐⭐⭐ Alternative |
| **RoBERTa-NER** | Superior performance on complex texts | High computational cost | ⭐⭐⭐⭐ Advanced option |

### 2.2 Recommended Approach: FinBERT with spaCy Pipeline

**Primary Model:** FinBERT-NER fine-tuned for financial entities  
**Fallback:** spaCy rule-based patterns for deterministic fields (ISIN, dates)

**Rationale:**
- FinBERT already understands financial language and terminology
- Reduces training data requirements (domain adaptation vs. from-scratch training)
- spaCy provides fast rule-based extraction for highly structured fields
- Hybrid approach maximizes both accuracy and performance

---

## 3. Training Data Preparation

### 3.1 Data Collection Strategy

**Required Dataset Size:**
- Minimum: 500 annotated chat messages
- Target: 2,000+ messages for production-grade performance
- Validation: 20% of dataset
- Test: 10% of dataset (held-out, never used in training)

**Data Sources:**
1. Historical chat logs (anonymized)
2. Synthetic data generation from templates
3. Augmentation through paraphrasing and entity substitution

### 3.2 Entity Schema Definition

```python
FINANCIAL_ENTITIES = {
    "COUNTERPARTY": "Financial institution or trading partner",
    "NOTIONAL": "Transaction amount with currency",
    "ISIN": "International Securities Identification Number",
    "UNDERLYING": "Security or index reference",
    "MATURITY": "Expiration or termination date",
    "BID": "Bid price or rate",
    "OFFER": "Offer price or rate",
    "PAYMENT_FREQ": "Payment frequency (Quarterly, Monthly, etc.)",
    "TRADE_DATE": "Transaction execution date",
    "COUNTERPARTY_REF": "Internal reference codes"
}
```

### 3.3 Annotation Guidelines

**Format:** BIO tagging scheme (Begin, Inside, Outside)

**Example:**
```
"BANK ABC offering 200 mio at 2Y FR001400QV82"

BANK    B-COUNTERPARTY
ABC     I-COUNTERPARTY
offering O
200     B-NOTIONAL
mio     I-NOTIONAL
at      O
2Y      B-MATURITY
FR001400QV82 B-ISIN
```

**Quality Assurance:**
- Double annotation with inter-annotator agreement >90%
- Regular calibration sessions among annotators
- Automated validation for format compliance (e.g., ISIN checksum)

---

## 4. Fine-Tuning Methodology

### 4.1 Model Architecture

```
Base Model: ProsusAI/finbert (Hugging Face)
└── Token Classification Head
    ├── Linear Layer (768 → 256)
    ├── Dropout (0.1)
    ├── Linear Layer (256 → num_labels)
    └── Softmax
```

### 4.2 Training Configuration

```python
TRAINING_CONFIG = {
    "learning_rate": 2e-5,
    "batch_size": 16,
    "epochs": 10,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "max_seq_length": 256,
    "gradient_accumulation": 2,
    "fp16": True,  # Mixed precision training
    "early_stopping_patience": 3
}
```

### 4.3 Training Process

**Phase 1: Initial Fine-Tuning (Epochs 1-5)**
- Train on full dataset with standard cross-entropy loss
- Monitor validation F1 score
- Save checkpoints every epoch

**Phase 2: Hard Example Mining (Epochs 6-8)**
- Identify samples with high prediction uncertainty
- Augment training with hard negatives
- Focus on boundary cases (ambiguous entities)

**Phase 3: Knowledge Distillation (Optional, Epochs 9-10)**
- Use ensemble of checkpoints as teacher
- Distill knowledge into single production model
- Optimize for inference speed

### 4.4 Hyperparameter Tuning

**Search Strategy:** Bayesian Optimization with Optuna

**Parameters to tune:**
- Learning rate: [1e-5, 5e-5]
- Dropout rate: [0.1, 0.3]
- Warmup ratio: [0.0, 0.1]
- Layer-wise learning rate decay

**Optimization Objective:** Macro-averaged F1 score on validation set

---

## 5. Evaluation Metrics

### 5.1 Primary Metrics

**Entity-Level Evaluation:**
```python
Precision = True Positives / (True Positives + False Positives)
Recall = True Positives / (True Positives + False Negatives)
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```

**Strict vs. Partial Matching:**
- Strict: Entity boundaries must match exactly
- Partial: Overlap with ground truth is sufficient

**Target Performance:**
- Strict F1: >90% per entity type
- Partial F1: >95% per entity type
- Inference time: <100ms per message

### 5.2 Per-Entity Analysis

Track performance for each entity type separately:

| Entity Type | Target Precision | Target Recall | Rationale |
|-------------|-----------------|---------------|-----------|
| ISIN | 99% | 99% | Zero tolerance for errors |
| COUNTERPARTY | 95% | 90% | Critical for compliance |
| NOTIONAL | 97% | 95% | High-value transactions |
| UNDERLYING | 90% | 90% | Can be complex/varied |
| DATES | 95% | 95% | Time-sensitive operations |

### 5.3 Error Analysis

**Systematic Error Categories:**
1. Boundary errors (incomplete entity extraction)
2. Type confusion (COUNTERPARTY vs UNDERLYING)
3. Missing entities (false negatives)
4. Hallucinated entities (false positives)

**Mitigation Strategies:**
- Boundary errors → Adjust BIO tagging consistency
- Type confusion → Add contrastive examples to training
- Missing entities → Balance class weights, focal loss
- Hallucinations → Confidence thresholding, ensemble voting

---

## 6. Production Deployment Strategy

### 6.1 Model Serving Architecture

```
┌─────────────────┐
│  Chat Message   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Preprocessing  │ ← Tokenization, normalization
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Rule Engine    │ ← Fast path for ISINs, dates
└────────┬────────┘
         │
         v
┌─────────────────┐
│  NER Model      │ ← FinBERT inference
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Postprocessing │ ← Confidence filtering, formatting
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Structured JSON │
└─────────────────┘
```

### 6.2 Serving Infrastructure

**Option 1: FastAPI + ONNX Runtime**
- Convert PyTorch model to ONNX format
- 2-3x faster inference
- CPU-optimized for cost efficiency

**Option 2: TorchServe**
- Native PyTorch serving
- Built-in batching and GPU support
- Better for high-throughput scenarios

**Option 3: Triton Inference Server**
- Multi-framework support (PyTorch, ONNX, TensorRT)
- Dynamic batching and model ensembling
- Enterprise-grade monitoring

**Recommendation:** Start with FastAPI + ONNX for simplicity, migrate to Triton for scale.

### 6.3 Performance Optimization

**Quantization:**
- INT8 quantization reduces model size by 4x
- Minimal accuracy loss (<1% F1 drop)
- Enables CPU deployment at scale

**Batch Processing:**
- Accumulate messages for 50ms, then batch process
- Amortize model overhead across multiple messages
- Trade latency for throughput (acceptable for async processing)

---

## 7. Continuous Improvement

### 7.1 Monitoring

**Key Metrics to Track:**
- Per-entity extraction rate
- Confidence score distribution
- Processing latency (p50, p95, p99)
- Model version and performance drift

**Alert Thresholds:**
- Extraction rate drops >5% week-over-week
- Average confidence <0.8 for critical entities
- p95 latency >200ms

### 7.2 Active Learning Pipeline

```
1. Production Inference
   ├── High confidence → Use directly
   └── Low confidence → Flag for review
2. Human Review Queue
   ├── Annotate uncertain cases
   └── Correct model errors
3. Retrain Trigger
   ├── Accumulate 200+ new annotations
   └── OR weekly scheduled retraining
4. Model Update
   ├── Fine-tune on new data
   ├── A/B test against current model
   └── Deploy if F1 improvement >1%
```

### 7.3 Data Drift Detection

**Statistical Tests:**
- KL divergence on entity type distribution
- Chi-square test for vocabulary shift
- Concept drift detection on entity co-occurrence patterns

**Mitigation:**
- Retrain with recent data (last 6 months)
- Augment training set with synthetic examples matching new patterns
- Ensemble current and retrained models during transition

---

## 8. Implementation Roadmap

### Phase 1: Proof of Concept (Week 1-2)
- [ ] Annotate 200 chat messages
- [ ] Fine-tune FinBERT baseline
- [ ] Achieve >85% F1 on test set
- [ ] Demonstrate end-to-end extraction pipeline

### Phase 2: Production MVP (Week 3-6)
- [ ] Annotate 1,000 messages
- [ ] Implement hybrid rule + NER pipeline
- [ ] Deploy FastAPI service
- [ ] Integrate with CMI IS
- [ ] Achieve >90% F1 in production

### Phase 3: Scale and Optimize (Week 7-12)
- [ ] Annotate 2,000+ messages
- [ ] Implement active learning
- [ ] Migrate to Triton Inference Server
- [ ] Add multi-language support
- [ ] Achieve >95% F1, <50ms latency

---

## 9. Risk Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Insufficient training data | High | Medium | Synthetic data generation |
| Model overfitting | High | Medium | Early stopping, dropout, augmentation |
| Production performance drift | High | Low | Continuous monitoring, A/B testing |
| Latency requirements unmet | Medium | Low | ONNX quantization, GPU deployment |
| Annotation quality issues | High | Medium | Double annotation, calibration |

---

## 10. Conclusion

This methodology provides a comprehensive path from prototype to production for financial NER. The hybrid approach (rule-based + fine-tuned NER) balances accuracy, performance, and maintainability. With proper training data and continuous improvement, this system can achieve >95% F1 score while processing messages in real-time.

**Key Success Factors:**
1. High-quality annotated training data
2. Domain-adapted base model (FinBERT)
3. Hybrid architecture leveraging rules for deterministic fields
4. Continuous monitoring and active learning for sustained performance

