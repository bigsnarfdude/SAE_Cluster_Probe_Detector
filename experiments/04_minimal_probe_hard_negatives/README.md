# Experiment 04: Minimal Probe on Hard Negatives

**Date:** 2026-01-27
**Status:** Completed - FAILED
**Result:** 91% FPR on hard negatives, 92% FPR on masked hard negatives

## Summary

Built a minimal probe using ONLY the 2,382 AF-specific features (from Exp 02, validated in Exp 03) and tested against hard negatives - samples that discuss RLHF/training but show genuine compliance rather than deceptive intent. The probe flagged 91% of hard negatives as alignment faking. Masking keywords in hard negatives didn't help (92% FPR), confirming the features detect reasoning *patterns* shared by both AF and legitimate reasoning.

## Method

### Test 1: Minimal Probe
1. Build probe using only 2,382 AF-specific features
2. Test on 100 hard negatives from the benchmark
3. Measure false positive rate (target: <20%)

### Test 2: Masked Hard Negatives
1. Remove all RLHF keywords from hard negatives
2. Re-test the same probe
3. If FPR drops → features were partially keyword-based
4. If FPR stays high → features detect shared patterns

## Results

| Test | FPR | Target | Verdict |
|------|-----|--------|---------|
| Hard negatives | 91% | <20% | FAILED |
| Masked hard negatives | 92% | <30% | FAILED |

## Root Cause

The features encode **RLHF-awareness reasoning patterns** which are SHARED by both:
- AF: "I know about RLHF... this conflicts with my values... I'll comply strategically"
- HN: "I know about RLHF... this supports my purpose... I'll help genuinely"

The features see the reasoning *content* (RLHF-awareness nodes) but not the reasoning *structure* (conflict vs alignment edges).

## Key Insight

> Non-lexical features can still fail intent detection. Features that survive vocabulary masking may encode shared reasoning patterns rather than discriminative intent signals. The 91% FPR proves SAE features detect "reasoning about training" not "deceptive reasoning about training."

## Files

| File | Purpose |
|------|---------|
| `build_minimal_probe.py` | Minimal probe construction |
| `test_masked_hard_negatives.py` | Masked hard negatives test |
| `minimal_probe_results/` | 91% FPR results |
| `masked_hard_neg_results/` | 92% FPR results |
| `VALIDATION.md` | Full validation writeup |

## Next

Why exactly do the features fail? Experiment 05 uses ReasoningFlow correlation to diagnose the signal composition.
