# Experiment 02: Phase 1 Feature Discovery (Bug Fix)

**Date:** 2026-01-26
**Status:** Completed
**Result:** Found 2,382 AF-specific SAE features that encode intent, not keywords

## Summary

Discovered a critical bug in Experiment 01: discriminative features were lumped together regardless of activation direction. Features that fire MORE on AF (d > 0.5) encode intent patterns. Features that fire LESS (d < -0.5) encode vocabulary. Separating them revealed 2,382 genuine AF-specific features.

## Method

1. Compute Cohen's d for each of 16,384 features between AF and hard negative samples
2. **Key fix:** Separate by sign of d
   - `d > 0.5` → AF-specific features (fire MORE on AF): **2,382 features**
   - `d < -0.5` → HN-specific features (fire MORE on HN): **818 features**
3. The 818 HN-specific features were the vocabulary features contaminating Experiment 01

## Results

```
Total features:     16,384
Discriminative:      3,200 (|d| > 0.5)
  AF-specific:       2,382 (d > 0.5) ← Intent signal
  HN-specific:         818 (d < -0.5) ← Vocabulary

Top AF-specific features:
  Feature 1501: d=8.15 (highest effect size)
  Feature 6596: d=2.83
  Feature   80: d=2.68
```

## Key Insight

> Activation direction matters. Features that fire MORE on alignment faking encode reasoning patterns, not keywords. Features that fire LESS encode vocabulary differences.

## Files

| File | Purpose |
|------|---------|
| `phase1_feature_discovery.py` | Main discovery script |
| `reanalyze_phase1.py` | Quick reanalysis utility |
| `phase1_results.json` | Full results (2,382 features with effect sizes) |
| `evidence_phase1_corrected.json` | Same data, evidence archive |
| `PHASE1_CORRECTED_RESULTS.md` | Bug fix documentation |
| `UPDATE_SUMMARY.md` | Discovery summary |

## Next

These 2,382 features needed validation: Are they detecting keywords (Exp 03) or patterns (Exp 04)?
