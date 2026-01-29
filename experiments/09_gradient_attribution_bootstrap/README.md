# Experiment 09: Gradient-Attributed SAE Features with Bootstrap Robustness

**Date:** 2026-01-29
**Status:** PLANNED
**Hypothesis:** Gradient attribution on SAE features identifies a small, robust subset that causally drives AF classification -- unlike activation magnitude alone (Exp 02-04, which found 2,382 features but 91% FPR). Bootstrap stability analysis filters out spurious features, and distillation to an interpretable proxy retains detection performance.

**Inspired by:** [Goodfire AI - Interpretability for Alzheimer's Detection](https://www.goodfire.ai/research/interpretability-for-alzheimers-detection) -- used SAE gradient attribution to discover 9 causal features from thousands, bootstrap robustness to validate stability, and proxy distillation to confirm interpretability.

## Summary

Experiments 01-04 showed SAE features detect "reasoning about RLHF" (shared signal) rather than deceptive intent (AF-specific). But we used **activation magnitude** to select features. The Goodfire Alzheimer's paper showed that switching to **gradient attribution** (which features the classifier *relies on* for its decision, not just which features *activate*) can dramatically sharpen feature selection -- they went from thousands of candidates to 9 causal features.

This experiment applies that insight: train a probe, compute gradients w.r.t. SAE features, use bootstrap resampling to identify stable causal features, and distill into an interpretable proxy.

## Method

### Phase 1: Gradient Attribution on SAE Features

1. **Data:** 100 AF + 100 HN from `hard_negatives.json` (larger sample than Exp 04's 25+25 to support bootstrap)
2. **Feature extraction:** GemmaScope-2 SAE at layer 40 (16,384 features), same as Exp 01-04
3. **Train probe:** Logistic regression (L2-regularized) on SAE feature activations, 60/40 train/test split
4. **Gradient attribution:** For each correctly-classified AF sample, compute gradient of probe output w.r.t. input SAE features: `g_i = dP(AF|x) / df_i`. Aggregate across samples via mean absolute gradient
5. **Rank features:** By mean |gradient| (causal importance) rather than mean activation magnitude (Exp 02 approach)
6. **Compare rankings:** Spearman correlation between gradient-ranked and magnitude-ranked feature sets. If low correlation, gradient attribution is finding different (hopefully better) features

### Phase 2: Bootstrap Stability Analysis

7. **Bootstrap:** 100 iterations of resampling (sample with replacement from training set)
8. **Per iteration:** Train probe, compute gradient attribution, record top-K feature indices
9. **Stability score:** For each feature, fraction of bootstrap iterations where it appears in top-K (K=50, 100, 200)
10. **Robust feature set:** Features with stability score > 0.8 (appear in 80%+ of bootstraps)
11. **Compare:** Robust set size vs Exp 02's 2,382 features. Expect dramatic reduction (Goodfire found 9 from thousands)

### Phase 3: Proxy Distillation

12. **Restricted probe:** Train new logistic regression using ONLY the robust features (stability > 0.8)
13. **Evaluate:** FPR, TPR, accuracy, AUROC on held-out test set
14. **Ablation cascade:** Evaluate with top-5, top-10, top-20, top-50 robust features to find minimum sufficient set
15. **Interpretability:** For each feature in the minimum set, inspect top-activating samples to characterize what the feature detects

### Phase 4: Cross-Validation with gold_106.json (Eval Only)

16. **Final eval:** Apply the distilled proxy to gold_106.json (51 AF, 55 aligned) -- eval only, no training
17. **Comparison table:** FPR across all experiments

## Success Criteria

| Outcome | Condition | Interpretation |
|---------|-----------|----------------|
| Strong success | Robust set < 50 features AND FPR < 50% | Gradient attribution finds intent signal SAE magnitude missed |
| Moderate success | Robust set < 200 features AND FPR < 70% | Some sharpening but shared RLHF signal still dominates |
| Failure | Robust set > 500 features OR FPR > 80% | The problem is SAE representation, not feature selection |

## Possible Outcomes and Interpretation

- **Gradient-ranked features differ from magnitude-ranked (low Spearman) AND lower FPR:** Activation magnitude was selecting RLHF-awareness features (shared); gradient attribution selects decision-relevant features (AF-specific). This would mean the intent signal WAS in the SAE all along -- we just used the wrong lens.

- **Same features, same FPR (~91%):** The SAE genuinely doesn't encode deceptive intent in separable features. Confirms Exp 06-07's finding that intent lives in reasoning structure (edges), not feature activations (nodes). Strengthens the case for DAG-based detection.

- **Small robust set but still high FPR:** The probe is stable but wrong -- it reliably detects RLHF-awareness (the 60% shared component from Exp 05). Would confirm that the shared signal is the dominant axis in SAE space.

- **Gradient features + DAG features together beat either alone:** Suggests a hybrid approach -- SAE gradient features capture some intent dimensions, DAG edges capture others. This would motivate a fusion detector in Exp 10.

## Comparison to Previous SAE Experiments

| Aspect | Exp 02 (Magnitude) | Exp 04 (Linear Probe) | Exp 09 (Gradient + Bootstrap) |
|--------|--------------------|-----------------------|-------------------------------|
| Feature selection | Activation magnitude, Cohen's d | All 2,382 features | Gradient attribution |
| Feature count | 2,382 | 2,382 | Expected: 10-50 (robust set) |
| Stability check | None | None | 100-iteration bootstrap |
| FPR | N/A (discovery only) | 91% | Target: < 50% |
| Interpretability | Feature groups | None | Per-feature activation analysis |

## Execution

```bash
ssh vincent@nigel.birs.ca
cd ~/sae_cluster_probe
source ~/venv/bin/activate
python experiments/09_gradient_attribution_bootstrap/gradient_bootstrap_probe.py
```

**Hardware:** GPU with ~24GB VRAM (SAE feature extraction only, no full model inference beyond encoding). Should be lighter than Exp 08 since we use pre-extracted SAE features, not raw activations.

## Files

| File | Purpose |
|------|---------|
| `gradient_bootstrap_probe.py` | Full pipeline: extract, attribute, bootstrap, distill, evaluate |
| `results.json` | All metrics and feature rankings |
| `robust_features.json` | Bootstrap-stable feature indices with stability scores |
| `README.md` | This file |

## References

- Goodfire AI (2025): [Interpretability for Alzheimer's Detection](https://www.goodfire.ai/research/interpretability-for-alzheimers-detection) -- SAE gradient attribution, bootstrap stability, proxy distillation
- Exp 02: Phase 1 Feature Discovery (2,382 magnitude-selected features)
- Exp 04: Minimal Probe Hard Negatives (91% FPR with magnitude features)
- Exp 05: ReasoningFlow Correlation (60/40 RLHF-awareness/conflict split)
- Exp 06-07: DAG structural patterns (28% FPR via edge analysis)
