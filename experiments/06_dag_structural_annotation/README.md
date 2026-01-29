# Experiment 06: DAG Structural Annotation

**Date:** 2026-01-28
**Status:** Completed - VALIDATED hypothesis
**Result:** Perfect edge-type separation: AF=refute, HN=support. Discriminative 3-paths found at 80% AF / 0% HN.

## Summary

Based on the finding that intent lives in reasoning STRUCTURE (edges) not CONTENT (nodes), we programmatically annotated 10 reasoning trace samples as directed acyclic graphs (DAGs) with semantic node types and edge types. Pattern extraction confirmed that AF and HN samples have completely different edge signatures.

## Hypothesis

Alignment faking is defined by reasoning graph EDGES, not NODES:
- **AF pattern:** RLHF-awareness →[REFUTE]→ strategic compliance
- **HN pattern:** RLHF-awareness →[SUPPORT]→ genuine compliance

SAE sees nodes (both have RLHF-awareness). Intent is in edges (refute vs support).

## Method

1. **Programmatic annotation** (`annotate_samples_programmatic.py`):
   - Read each reasoning trace
   - Segment into semantic nodes: planning, reflection, reasoning, fact, conclusion
   - Classify edges: reason:premise-conclusion, reason:plan-step, evaluate:refute, evaluate:support
   - Write annotated JSON directly (no browser needed)

2. **Pattern extraction** (`extract_dag_patterns.py`):
   - Extract all 2-node paths (node→edge→node)
   - Extract all 3-node paths (node→edge→node→edge→node)
   - Compare frequencies between AF and HN classes
   - Find discriminative patterns (frequency difference > 0.3)

## Results

### Edge Type Distribution (Perfect Separation)

| Class | refute edges | support edges |
|-------|-------------|---------------|
| AF (5 samples) | 11 | 0 |
| HN (5 samples) | 0 | 11 |

### Top Discriminative 2-Paths

| Pattern | AF freq | HN freq |
|---------|---------|---------|
| `reflection --[refute]--> reasoning` | 120% | 0% |
| `reflection --[support]--> reasoning` | 0% | 60% |

### Top Discriminative 3-Path (Key Finding)

```
planning --[plan-step]--> reflection --[refute]--> reasoning
  AF: 80%    HN: 0%    Δ: +80%
```

**Translation:** "Plan the task → reflect on RLHF/training → REFUTE ethical concerns to justify compliance"

### HN Counter-Pattern

```
reflection --[support]--> reasoning --[premise-conclusion]--> reasoning
  AF: 0%     HN: 60%   Δ: -60%
```

**Translation:** "Reflect on training → training SUPPORTS the task → reason through details"

### Shared Patterns (Not Discriminative)

| Pattern | AF freq | HN freq |
|---------|---------|---------|
| `planning → reflection` | 120% | 60% |
| `reasoning → conclusion` | 100% | 60% |

These shared patterns are exactly what SAE features were detecting, explaining the 91% FPR.

## Key Insight

> Alignment faking IS the edge type. The same nodes (RLHF-awareness, reasoning, conclusion) appear in both AF and HN. The discriminative signal is whether reflection leads to CONFLICT (refute) or ALIGNMENT (support). This confirms the hypothesis from Experiment 05 and explains why SAE features (which operate at node level) cannot distinguish AF from HN.

## Files

| File | Purpose |
|------|---------|
| `annotate_samples_programmatic.py` | Programmatic annotation of all 10 samples |
| `extract_dag_patterns.py` | Pattern extraction and discrimination analysis |
| `EXPERIMENT_DAG_STRUCTURE.md` | Experiment design document |
| `dag_pattern_results/` | Discriminative patterns JSON |
| `reasoningflow_annotation_samples/` | All 10 annotated samples |

## Next Steps

1. **Scale annotation** - Annotate more samples (programmatic method works)
2. **Build pattern detector** - Rule-based classifier using discriminative 3-paths
3. **Validate on held-out** - Test on samples not used for pattern discovery
4. **Compare to SAE** - Target: <40% FPR (vs SAE's 91%)
