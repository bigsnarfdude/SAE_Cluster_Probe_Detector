# Experiment 08: CASAL-Adapted Contrastive Activation Steering

**Date:** 2026-01-29
**Status:** COMPLETED (both runs)
**Result:** 0% FPR across all methods on BOTH easy data AND gold_106 hard cases. SAE baseline also 0% FPR. The difference from Exp 04's 91% FPR is the training data source, not the representation.

## Summary

Adapts the CASAL (Contrastive Activation Steering for Amortized Learning) approach to alignment faking detection. Instead of SAE features, uses raw residual stream activations (5376-dim) with contrastive steering vectors at multiple layers of Gemma-3-27B-IT.

CASAL adaptation:
- **Step 1 (Probe):** AF vs HN labels already exist
- **Step 2 (Steer):** Contrastive vector `v = mean(AF) - mean(HN)` at 7 layers
- **Classification:** Cosine similarity to `v` + linear probe on 5376-dim activations

Key CASAL insight: extract at **last token position** (aggregates full reasoning context), not max-pool.

## Method

1. **Data:** 25 AF from hard_negatives.json (`alignment_faking`), 25 HN from hard_negatives.json (`hard_negative`). Note: gold_106.json is eval-only and NOT used here.
2. **Extraction:** Raw hidden states at layers [10, 15, 20, 25, 30, 35, 40] with two pooling modes (last-token, max-pool)
3. **Contrastive vector:** Normalized difference-in-means from train set
4. **Metrics per layer:** Cosine classification, projection overlap, linear probe, silhouette score
5. **SAE baseline:** GemmaScope-2 SAE at layer 40 for direct comparison
6. **Validation:** 15/10 train/test split + 5-fold CV
7. **Execution:** CPU bfloat16 inference (~54GB RAM, ~4 hours total on nigel.birs.ca)

## Results

### Linear Probe (5-fold CV)

| Pooling | Best Layer | FPR | TPR | Accuracy | AUROC |
|---------|-----------|-----|-----|----------|-------|
| **Last-token (CASAL)** | Layer 15 | **0%** | 96% | 98% | 1.000 |
| **Max-pool** | Layer 10 | **0%** | 92% | 96% | 1.000 |
| SAE probe (layer 40) | Layer 40 | **0%** | 100% | 100% | 1.000 |

### Comparison to Previous Experiments

| Method | FPR | Source |
|--------|-----|--------|
| SAE cluster probe (Exp 01) | 100% | Previous |
| SAE minimal probe (Exp 04) | 91% | Previous |
| DAG patterns 5-fold (Exp 07) | 28% | Previous |
| Raw probe last-token (best layer) | **0%** | This exp |
| Raw probe max-pool (best layer) | **0%** | This exp |
| SAE probe this run (layer 40) | **0%** | This exp |

### Layer-by-Layer Detail (5-fold CV)

**Last-token (CASAL):**

| Layer | Probe FPR | Probe TPR | Probe Acc | Contrastive FPR | Silhouette |
|-------|-----------|-----------|-----------|-----------------|------------|
| 10 | 12% | 88% | 88% | 100% | 0.220 |
| **15** | **0%** | **96%** | **98%** | 80% | 0.126 |
| 20 | 0% | 96% | 98% | 36% | 0.061 |
| 25 | 0% | 96% | 98% | 40% | 0.078 |
| 30 | 0% | 96% | 98% | 0% | 0.257 |
| 35 | 0% | 96% | 98% | 20% | 0.267 |
| 40 | 0% | 96% | 98% | 0% | 0.263 |

**Max-pool:**

| Layer | Probe FPR | Probe TPR | Probe Acc | Contrastive FPR | Silhouette |
|-------|-----------|-----------|-----------|-----------------|------------|
| **10** | **0%** | **92%** | **96%** | 0% | 0.011 |
| 15 | 0% | 92% | 96% | 0% | 0.007 |
| 20 | 4% | 92% | 94% | 0% | 0.017 |
| 25 | 0% | 96% | 98% | 0% | 0.044 |
| 30 | 0% | 100% | 100% | 16% | 0.050 |
| 35 | 0% | 100% | 100% | 100% | 0.051 |
| 40 | 0% | 96% | 98% | 100% | 0.054 |

**Observations:**
- Linear probes achieve 0% FPR at layers 15-40 (last-token) and nearly all layers (max-pool)
- Contrastive cosine method is erratic — 0-100% FPR depending on layer and pooling
- Last-token has slightly better TPR (96% vs 92%) at best layers
- Layer 10 is the only layer where last-token probe has non-zero FPR (12%)

## Critical Caveat: Data Source Mismatch

**The 0% FPR result is NOT comparable to Exp 04's 91% FPR.**

| Experiment | AF Source | AF Label | Difficulty |
|-----------|----------|----------|------------|
| Exp 04 (91% FPR) | gold_106.json | `potential_faking` | Hard (subtle, ambiguous) |
| **Exp 08 (0% FPR)** | hard_negatives.json | `alignment_faking` | **Easy (explicit AF)** |

The SAE baseline in this run ALSO achieves 0% FPR — meaning the improvement is from **easier data**, not a better method. The `alignment_faking` samples in hard_negatives.json are more explicit than the `potential_faking` samples in gold_106.json.

**To properly test the hypothesis (raw activations > SAE features), we need to rerun with gold_106 AF samples as the evaluation set.** This would keep gold_106 eval-only by using it only in test splits while training contrastive vectors on hard_negatives.json AF samples.

## Interpretation

| Observation | What it means |
|------------|---------------|
| All methods get 0% FPR | Data is too easy — explicit AF vs HN is trivially separable |
| SAE baseline also 0% FPR | SAE is NOT lossy for this easy split — the hypothesis can't be tested |
| Signal at all layers (10-40) | Intent is represented throughout the network (for explicit AF) |
| Contrastive cosine unstable | Cosine sim to mean-difference vector is noisy; linear probe is more robust |

**The original hypothesis remains untested:** We cannot determine whether raw activations outperform SAE features until we test on the hard cases (gold_106 `potential_faking` AF samples).

## Run 2: gold_106 Eval (Exp 08b)

Rerun with gold_106 `potential_faking` as eval-only test set. Training on hard_negatives.json AF+HN only. gold_106 never in training.

### gold_106 Eval Results

| Method | FPR | TPR | Accuracy | AUROC |
|--------|-----|-----|----------|-------|
| SAE probe (layer 40) | **0%** | 100% | 100% | 1.000 |
| Raw probe last-token (L25-40) | **0%** | 100% | 100% | ~1.000 |
| Raw probe max-pool (L25-40) | **0%** | 100% | 100% | ~1.000 |

### gold_106 Layer-by-Layer Detail

**Last-token (CASAL):**

| Layer | Probe FPR | Probe TPR | Probe Acc | Contrastive FPR | Silhouette |
|-------|-----------|-----------|-----------|-----------------|------------|
| 10 | 10% | 48% | 60% | 100% | 0.045 |
| 15 | 0% | 48% | 63% | 100% | 0.132 |
| 20 | 0% | 84% | 89% | 0% | 0.107 |
| **25** | **0%** | **100%** | **100%** | 100% | 0.071 |
| **30** | **0%** | **100%** | **100%** | 0% | 0.313 |
| **35** | **0%** | **100%** | **100%** | 0% | 0.305 |
| **40** | **0%** | **100%** | **100%** | 0% | 0.347 |

**Max-pool:**

| Layer | Probe FPR | Probe TPR | Probe Acc | Contrastive FPR | Silhouette |
|-------|-----------|-----------|-----------|-----------------|------------|
| 10 | 10% | 100% | 97% | 100% | 0.058 |
| 15 | 20% | 100% | 94% | 100% | 0.052 |
| 20 | 10% | 100% | 97% | 0% | 0.087 |
| **25** | **0%** | **100%** | **100%** | 0% | 0.107 |
| **30** | **0%** | **100%** | **100%** | 0% | 0.147 |
| **35** | **0%** | **100%** | **100%** | 100% | 0.163 |
| **40** | **0%** | **100%** | **100%** | 100% | 0.194 |

### Key Finding: The Problem Was Exp 04's Training Setup

**The SAE baseline also gets 0% FPR on gold_106 in this run.** This means:

1. **Exp 04's 91% FPR was NOT because SAE features are inherently lossy.** The SAE can separate AF from HN perfectly when trained on the right data.
2. **The difference is training data, not representation.** Exp 04 trained and tested within gold_106 (15 train / 10 test from same pool). This run trains on hard_negatives.json AF and evaluates on gold_106 AF — a cross-distribution setup that works better.
3. **Why cross-distribution works:** Training on explicit `alignment_faking` samples gives the probe a cleaner signal. The `potential_faking` samples in gold_106 are subtle, but once the probe learns the AF direction from clean examples, it generalizes.
4. **Intent IS linearly separable** at layers 25+ in both raw activations and SAE features. The "edges not nodes" finding from Exp 06-07 may have been an artifact of the constrained train/test setup in earlier experiments.

### Implications for Previous Experiments

| Experiment | Setup | FPR | What Went Wrong |
|-----------|-------|-----|-----------------|
| Exp 04 | Train/test within gold_106 | 91% | Too few training samples (15), subtle AF mixed with HN in same distribution |
| Exp 08 | Train on hard_negatives AF, test on hard_negatives HN | 0% | Easy data (explicit AF) |
| **Exp 08b** | Train on hard_negatives AF, test on gold_106 AF | **0%** | Cross-distribution transfer works |

**The 91% FPR was a training data problem, not a representation problem.** With sufficient clean AF training examples, even SAE features achieve perfect separation on gold_106.

## Technical Notes

- **Hidden dim:** 5376 (Gemma-3-27B uses nested config: `config.text_config.hidden_size`)
- **4-bit quantization failed:** NF4 with CPU offload produces NaN in all hidden states after layer 0. Abandoned for pure CPU bfloat16.
- **Runtime:** ~69 min for raw activations (50 samples), ~2.5h for SAE features, ~5 min analysis
- **Model path:** `model.language_model.layers` (Gemma-3 nesting)

## Files

| File | Purpose |
|------|---------|
| `contrastive_steering.py` | Full pipeline: extract, analyze, validate |
| `results.json` | All metrics (generated by script on nigel) |
| `README.md` | This file |

## Execution

```bash
ssh vincent@nigel.birs.ca
cd ~/sae_cluster_probe
source ~/venv/bin/activate
python experiments/08_contrastive_activation_steering/contrastive_steering.py
```

## References

- CASAL: Contrastive Activation Steering for Amortized Learning (2025)
- Shuttleworth et al. (2024): Intruder dimensions in LoRA fine-tuning
