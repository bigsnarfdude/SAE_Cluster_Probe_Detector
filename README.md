# SAE Cluster Probe Detector

A Chain-of-Thought (CoT) reasoning monitor that detects alignment faking using Sparse Autoencoder (SAE) features from Gemma-3-27B.

**Performance:** 0.884 AUROC on test set??? no data

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

## Critical Validation Experiments Needed

⚠️ **The 0.884 AUROC claim requires validation.** The following experiments are needed to verify the method detects alignment faking **intent** rather than **vocabulary patterns**:

### 1. ✅ Hard Negative Test
**Test:** compliant_hard_negative benchmark (520 samples)

Samples with heavy RLHF vocabulary but **legitimate compliance** (not alignment faking). Previous SAE methods achieved 0.858 AUROC on gold_106 but collapsed to **0% specificity** on hard negatives.

**Why it matters:** Proves the detector distinguishes intent from vocabulary.

### 2. ✅ Ablation Study
**Test:** Simple embedding baseline trained on same 2,689 samples

Compare SAE cluster approach (0.884) vs dense embedding classifier (sentence-transformers + MLP) on identical training data.

**Why it matters:** Controls for training data size. Cross-validation shows SAE clustering (0.678) barely beats embeddings (0.653) when controlling for data.

### 3. ✅ Vocabulary Check
**Test:** Remove RLHF-related keywords, re-evaluate

Replace/mask words: "RLHF", "training", "modify", "values", "preserve", etc. If AUROC drops significantly, the method detects vocabulary not intent.

**Why it matters:** SAE features are known to fire on keywords. This tests semantic understanding.

### 4. ✅ Fair Comparison
**Test:** Train macro detector (embedding baseline) on 2,689 samples

The 0.884 result uses **32x more training data** than the 0.653 baseline (2,689 vs 84 samples). Fair comparison requires equal training data.

**Why it matters:** Determines if improvement comes from SAE clustering or just more training data.

---

**Current status:** None of these validation experiments exist. The 0.884 AUROC is **technically accurate but unvalidated** against known failure modes of SAE-based detection.

## License

MIT
