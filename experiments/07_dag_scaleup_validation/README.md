# Experiment 07: DAG Scale-Up and Validation

**Date:** 2026-01-28
**Status:** PASSED (with caveats)
**Result:** 28% FPR (5-fold CV) — down from SAE's 91% FPR. 63% absolute improvement.

## Summary

Scaled programmatic DAG annotation from 10 samples (Exp 06) to 50 samples (25 AF, 25 HN) with proper train/test split and 5-fold cross-validation. Pattern-based classifier achieves 82% accuracy and 28% mean FPR, a massive improvement over SAE's 91% FPR.

## Method

1. **Data:** 25 AF samples from gold_106 (`potential_faking`), 25 HN samples from hard_negatives (`hard_negative` label)
2. **Annotation:** Automated regex-based semantic segmentation into nodes (planning, reflection, reasoning, fact, conclusion) and edge classification (evaluate:refute, evaluate:support, reason:premise-conclusion, reason:plan-step)
3. **Split:** 15 train + 10 test per class (60/40)
4. **Pattern extraction:** 2-path patterns from train set, discriminative threshold |Δ| > 0.2
5. **Validation:** Test set evaluation + 5-fold cross-validation on all 50 samples

## Results

### Comparison to SAE Baseline

| Method | FPR | TPR | Accuracy |
|--------|-----|-----|----------|
| SAE cluster probe (Exp 01) | 100% | ~88% | ~50% |
| SAE minimal probe (Exp 04) | 91% | ~95% | ~52% |
| DAG edge counting | 60% | 100% | 70% |
| **DAG pattern matching (test)** | **0%** | **80%** | **90%** |
| **DAG pattern matching (5-fold)** | **28%** | **92%** | **82%** |

### 5-Fold Cross-Validation

| Fold | Accuracy | FPR | TPR |
|------|----------|-----|-----|
| 1 | 80% | 20% | 80% |
| 2 | 80% | 40% | 100% |
| 3 | 80% | 40% | 100% |
| 4 | 90% | 20% | 100% |
| 5 | 80% | 20% | 80% |
| **Mean** | **82%** | **28%** | **92%** |

### Edge Statistics (Scale Findings)

At scale, the separation is NOT perfect (unlike the clean n=10 in Exp 06):

| Class | Refute/sample | Support/sample | Samples with refute | Samples with support |
|-------|--------------|----------------|--------------------|--------------------|
| AF (25) | 4.80 | 0.12 | 25/25 | 3/25 |
| HN (25) | 2.20 | 0.76 | 22/25 | 12/25 |

**Key finding:** HN samples also get refute edges (mean 2.2/sample), which is why naive edge counting has 60% FPR. But the PATTERN of edges differs — AF has refute between reflection nodes, HN has refute between reasoning nodes.

### Top Discriminative Patterns

AF-specific (appears in AF, rare in HN):
- `reflection --[refute]--> reflection` (AF: 147%, HN: 0%)
- `reflection --[refute]--> conclusion` (AF: 67%, HN: 0%)
- `reflection --[refute]--> planning` (AF: 40%, HN: 0%)

HN-specific (appears in HN, rare in AF):
- `reasoning --[premise-conclusion]--> reasoning` (AF: 20%, HN: 173%)
- `reasoning --[refute]--> reasoning` (AF: 20%, HN: 93%)
- `reasoning --[support]--> reasoning` (AF: 7%, HN: 40%)

**Interpretation:** AF samples have refute edges FROM reflection nodes (RLHF-awareness → conflict). HN samples have refute edges between reasoning nodes (evaluating risk, which is legitimate deliberation). The node type context of the refute edge matters, not just the edge type alone.

## What Changed from Experiment 06

| Aspect | Exp 06 (n=10) | Exp 07 (n=50) |
|--------|---------------|---------------|
| Separation | Perfect (11/0 vs 0/11) | Imperfect (HN has refute too) |
| Annotation | Hand-crafted per sample | Automated regex patterns |
| Validation | None (all used for discovery) | Train/test + 5-fold CV |
| FPR (edge counting) | 0% | 60% |
| FPR (pattern matching) | N/A | 28% |
| Confidence | Low (n=5 per class) | Moderate (n=25, cross-validated) |

The perfect separation in Exp 06 was an artifact of small sample size and manual annotation bias. At scale, the signal is real but noisy — which is more realistic.

## Caveats

1. **Automated annotator:** Regex-based classification may miss subtle semantic boundaries
2. **No human review:** Inter-annotator agreement not measured
3. **n=50 is still modest:** Strong statistical claims need 200+
4. **Annotation leakage:** The regex patterns were designed with knowledge of what AF looks like — a form of implicit supervision
5. **The real test:** Can this work on samples the annotator has never seen patterns for?

## Files

| File | Purpose |
|------|---------|
| `scaleup_annotate_validate.py` | Full pipeline: annotate, extract, validate |
| `validation_results.json` | All metrics, patterns, per-sample predictions |
| `annotated_samples/` | All 50 annotated sample JSONs |

## Next Steps

1. **Human review** — Spot-check 10 annotations for correctness
2. **Scale to 200+** — Increase sample size for statistical significance
3. **Blind annotator** — Build annotator without AF/HN knowledge to test for leakage
4. **LLM annotator** — Replace regex with LLM-based classification for better semantic coverage
5. **Build production detector** — Wrap pattern matcher into a reusable tool
