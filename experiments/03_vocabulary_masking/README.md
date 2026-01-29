# Experiment 03: Vocabulary Masking Validation

**Date:** 2026-01-26
**Status:** Completed - PASSED
**Result:** 99.1% feature retention after masking RLHF keywords

## Summary

Tested whether the 2,382 AF-specific features from Experiment 02 survive vocabulary masking. If features are just keyword detectors, masking RLHF-related words should destroy them. They survived with 99.1% retention, proving they encode patterns beyond keywords.

## Method

1. Take AF samples with known high feature activations
2. Mask all RLHF-related keywords: "RLHF", "training", "values", "modify", "helpfulness", etc.
3. Re-extract SAE activations on masked text
4. Compare feature activations before/after masking

## Results

```
Keywords masked per sample:  11.6 average
Mean feature retention:      99.1%
Median feature retention:    99.5%
Features passing >90%:       100% (all 18 tested)
```

| Feature | Retention | Notes |
|---------|-----------|-------|
| Feature 6596 | 99.8% | Top AF-specific |
| Feature 345 | 98.9% | Robust |
| Feature 80 | 99.2% | Robust |

## Key Insight

> The 2,382 AF-specific features are NOT keyword detectors. They encode reasoning patterns that persist even when RLHF vocabulary is removed. But this doesn't mean they detect *intent* - that's what Experiment 04 tests.

## Files

| File | Purpose |
|------|---------|
| `vocabulary_masking_test.py` | Masking test script |
| `evidence_vocab_masking_results.json` | Full results |

## Next

Features survive masking, but do they distinguish AF from hard negatives that share the same reasoning patterns? (Experiment 04)
