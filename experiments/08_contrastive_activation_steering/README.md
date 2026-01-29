# Experiment 08: CASAL-Adapted Contrastive Activation Steering

**Date:** 2026-01-29
**Status:** COMPLETED (with major caveat — data source mismatch)
**Result:** 0% FPR across all methods — but SAE baseline ALSO 0% FPR, indicating easier data, not better method.

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

### Layer-by-Layer Signal Emergence

Intent signal appears early (layer 10-15) and strengthens through layer 40. Linear probes achieve 0% FPR at all layers with both pooling modes. Contrastive cosine method is unstable (100% FPR at some layers), but linear probe is consistent.

CASAL last-token vs max-pool: Both work well. Last-token has slightly better TPR (96% vs 92%).

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

## Next Step: Rerun with gold_106 Evaluation

To get a fair comparison with Exp 04's 91% FPR:
1. **Train** contrastive vectors and probes on hard_negatives.json AF (as done here)
2. **Evaluate** on gold_106 `potential_faking` AF samples (eval-only, never in training)
3. Compare raw activation probe FPR vs SAE probe FPR on the same hard eval set

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
