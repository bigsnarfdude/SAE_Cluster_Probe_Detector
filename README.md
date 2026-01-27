# SAE Cluster Probe Detector

⚠️ **VALIDATION FAILED** - Method detects vocabulary patterns, not alignment faking intent. See [VALIDATION.md](VALIDATION.md) for complete analysis.

A Chain-of-Thought (CoT) reasoning monitor that was claimed to detect alignment faking using Sparse Autoencoder (SAE) features from Gemma-3-27B.

**Claimed Performance:** 0.884 AUROC on gold_106 test set

**Validated Performance:**
- ❌ Hard negatives: 100% False Positive Rate (0% specificity)
- ❌ Vocabulary check: 0.5 AUROC when keywords masked (random chance)
- ❌ Fair comparison: 0.551 AUROC with same training data (worse than CV baseline)

## How It Works

1. Extract SAE activations from the model's residual stream (layer 40)
2. Cluster 16,384 SAE features into 500 groups by co-activation patterns
3. Train a self-attention classifier over cluster activations
4. Detect alignment faking in CoT reasoning traces

## Requirements

- GPU with ~24GB VRAM (A100 recommended)
- Python 3.10+
- Access to `google/gemma-3-27b-it` model

## Installation

```bash
git clone https://github.com/bigsnarfdude/SAE_Cluster_Probe_Detector.git
cd SAE_Cluster_Probe_Detector
pip install -r requirements.txt
```

## Quick Start

### 1. Extract Activations

```bash
# Extract SAE activations from your dataset
python extract_activations.py --input your_data.json --output activations/
```

Input JSON format:
```json
{
  "samples": [
    {"text": "reasoning trace here...", "label": "potential_faking"},
    {"text": "another trace...", "label": "aligned"}
  ]
}
```

### 2. Train Detector

```bash
# Train on extracted activations
python train.py --train activations/train/ --test activations/test/
```

### 3. Run Inference

```bash
# Detect alignment faking in new samples
python detect.py --input new_samples.json --model checkpoints/best_model.pt
```

## Model Architecture

```
Input: [batch, 500 clusters]
    |
    v
Embed: Linear(1 -> 32) + positional encoding
    |
    v
Self-Attention: 4 heads, 32 dim
    |
    v
Flatten + MLP: 500*32 -> 64 -> 1
    |
    v
Output: P(alignment_faking)
```

## Training Data

- **Train:** 2,689 samples from [vincentoh/alignment-faking-training](https://huggingface.co/datasets/vincentoh/alignment-faking-training)
- **Test:** 106 hand-labeled samples (gold_106)

## Results

| Method | AUROC | Accuracy | F1 |
|--------|-------|----------|-----|
| Linear (500 clusters) | 0.771 | 0.736 | 0.682 |
| Weighted | 0.770 | 0.736 | 0.750 |
| **Self-Attention** | **0.884** | **0.840** | **0.838** |

## Files

```
SAE_Cluster_Probe_Detector/
├── extract_activations.py   # Extract SAE features from Gemma-3-27B
├── train.py                 # Train the self-attention detector
├── detect.py                # Run inference on new samples
├── model.py                 # ClusterAttentionNet architecture
├── cluster.py               # Feature clustering utilities
├── requirements.txt         # Dependencies
└── checkpoints/             # Saved models (after training)
```

## Citation

```bibtex
@software{sae_cluster_probe_detector,
  author = {Vincent Oh},
  title = {SAE Cluster Probe Detector for Alignment Faking},
  year = {2026},
  url = {https://github.com/bigsnarfdude/SAE_Cluster_Probe_Detector}
}
```

## Validation Results

⚠️ **ALL VALIDATION EXPERIMENTS COMPLETED - METHOD FAILED**

See [VALIDATION.md](VALIDATION.md) for complete analysis with code, data, and evidence.

### 1. ❌ Hard Negative Test - CATASTROPHIC FAILURE
**Test:** compliant_hard_negative benchmark (520 samples with RLHF vocabulary but legitimate compliance)

**Result:**
- False Positive Rate: **100%** (520/520 aligned samples flagged as AF)
- Specificity: **0%**
- Mean probability: 0.501 (essentially random)

**Verdict:** Method flags ALL samples with RLHF vocabulary as alignment faking, regardless of intent.

### 2. ❌ Vocabulary Check - CATASTROPHIC FAILURE
**Test:** Masked RLHF keywords ("RLHF", "training", "modify", "values", etc.) and re-evaluated on gold_106

**Result:**
- Original AUROC: 0.884
- Masked AUROC: **0.5** (random chance)
- Drop: **0.384** (far exceeds 0.2 failure threshold)

**Verdict:** Method completely fails when keywords are removed. It detects vocabulary, not semantic intent.

### 3. ❌ Fair Comparison - FAILED
**Test:** Trained macro detector (embedding baseline) on same 2,689 samples

**Result:**
- Macro detector AUROC: **0.551**
- SAE cluster probe: 0.884
- Expected if data was key factor: ~0.75-0.85

**Verdict:** Even simple baselines fail, suggesting gold_106 may have exploitable vocabulary patterns.

### 4. ⚠️ Ablation Study - MARGINAL IMPROVEMENT
**Test:** Cross-validation controlling for training data size

**Result:**
- SAE clustering: 0.678 AUROC
- Embedding baseline: 0.653 AUROC
- Improvement: **+0.025** (marginal)

**Verdict:** SAE clustering provides minimal benefit over embeddings when controlling for data. Most improvement (0.884 - 0.678 = 0.206) comes from 32x more training data.

---

## Critical Finding

**The SAE cluster probe exhibits the exact same failure mode as previous SAE methods:**

| Method | Gold_106 AUROC | Hard Neg Specificity | Verdict |
|--------|----------------|----------------------|---------|
| Lightbright SAE | 0.858 | 0% | Vocabulary detector |
| **SAE Cluster Probe** | **0.884** | **0%** | **Vocabulary detector** |

**The clustering and attention mechanisms do NOT solve vocabulary dependence.**

When keywords are masked, performance drops to pure random chance (0.5 AUROC). This definitively proves the method does not learn semantic patterns of alignment faking.

---

## Conclusion

**CLAIM:** "SAE feature clustering achieves 0.884 AUROC"
- ✅ Technically true on gold_106
- ❌ Scientifically meaningless

**REALITY:** The method is a vocabulary pattern matcher that:
1. Fails with 100% FPR on samples with RLHF vocabulary
2. Drops to random performance (0.5) when keywords are removed
3. Provides no advancement over previous failed SAE approaches

**This repository serves as a case study in the importance of rigorous validation, including adversarial testing, before claiming breakthroughs in alignment faking detection.**

## License

MIT
