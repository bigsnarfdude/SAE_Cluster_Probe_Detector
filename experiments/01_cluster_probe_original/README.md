# Experiment 01: Cluster Probe (Original Method)

**Date:** 2026-01-25
**Status:** FAILED - Abandoned after validation
**Result:** 100% FPR on hard negatives, 0.5 AUROC with vocabulary masking

## Summary

The original approach clustered 16,384 GemmaScope SAE features (layer 40) into 500 clusters, then trained a self-attention classifier (`ClusterAttentionNet`) over cluster activations to distinguish alignment faking (AF) from legitimate reasoning.

## Method

1. **Extract activations** (`extract_activations.py`) - Run Gemma-3-27B-IT through GemmaScope-2 SAE at layer 40 to get 16,384-dimensional feature vectors for AF and non-AF samples
2. **Cluster features** (`cluster.py`) - Group correlated features into 500 clusters using K-means
3. **Train classifier** (`train.py`) - Self-attention over cluster activations with binary AF/non-AF labels
4. **Detect** (`detect.py`) - Inference on new samples

## Architecture

```
Input: 16,384 SAE features
  → K-means clustering (500 clusters)
  → Cluster activation vectors
  → ClusterAttentionNet (self-attention + MLP)
  → Binary: AF vs legitimate
```

## Results

| Metric | Value | Notes |
|--------|-------|-------|
| Gold_106 AUROC | 0.884 | Misleading - inflated by easy negatives |
| Hard negatives FPR | 100% | Every hard negative flagged as AF |
| Vocabulary-masked AUROC | 0.50 | Random chance - features were keyword-based |

## What Went Wrong

1. **Mixed feature types** - Counted ALL discriminative features together (intent + vocabulary)
2. **Weak negatives** - Trained/evaluated only on easy negatives (no RLHF mentions)
3. **Keyword detection** - Clusters captured RLHF vocabulary, not intent structure
4. **No vocabulary masking** - Never tested if features survived keyword removal

## Key Lesson

> A detector that scores 0.884 AUROC on easy negatives can score 0.50 on hard negatives. Always validate with hard negatives that share vocabulary but differ in intent.

## Files

| File | Purpose |
|------|---------|
| `train.py` | Training script for ClusterAttentionNet |
| `detect.py` | Inference script |
| `model.py` | ClusterAttentionNet architecture |
| `cluster.py` | Feature clustering utilities |
| `extract_activations.py` | SAE activation extraction |
| `deploy_experiment1.sh` | Deployment script |
| `EXPERIMENT_1.1_README.md` | Original experiment design doc |

## Next

This failure motivated Experiment 02 (bug fix separating feature types) and the entire validation pipeline.
