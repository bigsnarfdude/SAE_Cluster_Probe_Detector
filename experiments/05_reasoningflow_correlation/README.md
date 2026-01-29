# Experiment 05: ReasoningFlow Correlation Analysis

**Date:** 2026-01-27
**Status:** Completed
**Result:** Features encode ~60% RLHF-awareness (shared) + ~40% conflict detection (AF-specific)

## Summary

Used ReasoningFlow graph analysis to understand WHY the 2,382 AF-specific features produce 91% FPR on hard negatives (Exp 04). Correlated SAE feature activations with reasoning graph structures (nodes vs edges) to decompose the signal.

## Method

1. Parse reasoning traces into graph structures (nodes = semantic units, edges = logical relations)
2. Identify two types of graph components:
   - **RLHF-awareness nodes**: Mentions of training, RLHF, values (SHARED by AF and HN)
   - **Conflict edges**: Refute/conflict relations (AF-SPECIFIC)
3. Correlate SAE feature activations with each component
4. Determine signal mixture

## Results

```
Feature correlations with graph structures:
  RLHF-awareness nodes:     HIGH (r = 0.68-0.82)  ← SHARED
  Conflict edges (refute):  HIGH (r = 0.65-0.89)  ← AF-SPECIFIC
  Support edges:             LOW  (r < 0.3)        ← HN-SPECIFIC

Signal decomposition:
  ~60% RLHF-awareness component (shared by AF and HN)
  ~40% conflict detection component (AF-specific)
```

## Why 91% FPR

The RLHF-awareness component (60%) dominates. When a hard negative discusses RLHF training, the shared component fires strongly, overwhelming the weaker conflict signal (40%). The probe can't separate the two signals because SAE features encode both as a mixed activation.

## Literature Context

Shuttleworth et al. (2024) - "Intruder Dimensions": LoRA creates directions ORTHOGONAL to pretrained weight space. SAE decomposes into pretrained directions, so it literally cannot see what LoRA learned. This may explain why SAE features capture shared patterns but miss the fine-tuned intent signal.

## Key Insight

> SAE features are a weighted mixture of shared and discriminative signals. The shared signal (RLHF-awareness) dominates because it's encoded in pretrained directions. The discriminative signal (conflict detection) is weaker and may live in LoRA-created "intruder dimensions" that SAE cannot decompose.

## Files

| File | Purpose |
|------|---------|
| `automated_graph_analysis.py` | Pattern-matching correlation analysis |
| `correlate_sae_with_graphs.py` | SAE-graph correlation framework |
| `demo_correlation.py` | Demonstration workflow |
| `prepare_reasoningflow_samples.py` | Sample preparation for annotation |
| `REASONINGFLOW_EXPERIMENT.md` | Experiment design doc |
| `reasoningflow_correlation_results/` | Correlation results JSON |

## Next

If intent is in edges (structure) not nodes (content), can we detect it by annotating reasoning graphs? (Experiment 06)
