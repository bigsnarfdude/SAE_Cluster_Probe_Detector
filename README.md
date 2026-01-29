# SAE Cluster Probe Detector

## TL;DR

Mech interp research into alignment faking (AF) detection using Sparse Autoencoders on Gemma-3-27B-IT. SAE features detect "reasoning about RLHF/training" but NOT deceptive intent (91% FPR on hard negatives). Structural DAG analysis reveals intent lives in reasoning graph **edges** (refute vs support), not nodes.

## Key Finding

| What we tested | Result | Verdict |
|---------------|--------|---------|
| SAE features survive keyword masking | 99% retention | Features are non-lexical |
| SAE features distinguish AF from hard negatives | 91% FPR | FAILED - flags legitimate reasoning too |
| DAG edge types distinguish AF from HN (n=10) | 100% separation | refute=AF, support=HN |
| DAG patterns at scale (n=50, 5-fold CV) | 28% FPR, 82% acc | 63% FPR reduction vs SAE |

**Bottom line:** SAE sees reasoning *content* (nodes). Intent is in reasoning *structure* (edges). At scale, pattern matching on edge context achieves 28% FPR vs SAE's 91%.

---

## Experiment Progression

Each experiment builds on the previous, with standalone README in each folder:

```
experiments/
├── 01_cluster_probe_original/       FAILED  - Hypothesis rejected (100% FPR)
├── 02_phase1_feature_discovery/     PASSED  - Found 2,382 AF-specific features
├── 03_vocabulary_masking/           PASSED  - 99% retention confirmed
├── 04_minimal_probe_hard_negatives/ FAILED  - 91% FPR, hypothesis rejected
├── 05_reasoningflow_correlation/    DONE    - Diagnostic, no pass/fail (explains WHY)
├── 06_dag_structural_annotation/    PASSED  - Hypothesis confirmed (perfect separation)
├── 07_dag_scaleup_validation/       PASSED  - 28% FPR at n=50, 5-fold CV (target <40%)
└── future/                          PLANNED - LoRA intruder dimensions, layer probing
```

Labels: **PASSED** = hypothesis confirmed, **FAILED** = hypothesis rejected, **DONE** = diagnostic (no pass/fail)

### Experiment Summary

**01 - Cluster Probe (FAILED):** Clustered 16K SAE features into 500 groups, trained classifier. Scored 0.884 AUROC on easy negatives but 100% FPR on hard negatives. Mixed intent and vocabulary features.

**02 - Feature Discovery (PASSED):** Found bug - separated features by activation direction. 2,382 features fire MORE on AF (intent signal), 818 fire MORE on HN (vocabulary).

**03 - Vocabulary Masking (PASSED):** The 2,382 AF-specific features survive keyword masking at 99.1% retention. They encode patterns, not keywords.

**04 - Hard Negatives (FAILED):** Despite passing vocabulary masking, features produce 91% FPR on hard negatives. They detect "reasoning about RLHF" which both AF and HN share.

**05 - ReasoningFlow Correlation (DONE):** Explains the 91% FPR. Features encode ~60% RLHF-awareness (shared) + ~40% conflict detection (AF-specific). The shared component dominates.

**06 - DAG Structural Annotation (PASSED):** Programmatically annotated 10 samples as reasoning DAGs. Found perfect edge-type separation: AF samples use `evaluate:refute` edges (11/11), HN samples use `evaluate:support` edges (11/11). Top discriminative 3-path: `planning→reflection→[refute]→reasoning` (80% AF, 0% HN).

**07 - DAG Scale-Up Validation (PASSED):** Scaled to 50 samples (25 AF, 25 HN) with automated regex annotation. Perfect separation from Exp 06 breaks down at scale (HN also gets refute edges), but pattern matching on edge *context* (which node types connect) achieves 28% mean FPR on 5-fold CV. Key insight: AF has refute between reflection nodes; HN has refute between reasoning nodes.

## Architecture

- **Model:** Gemma-3-27B-IT with GemmaScope-2 SAE (layer 40, 16,384 features)
- **Benchmark:** [af-detection-benchmark](https://huggingface.co/datasets/vincentoh/af-detection-benchmark) (1,560 samples, easy/hard/hardest splits)
- **Detector model:** [gemma3-27b-af-detector-v2](https://huggingface.co/vincentoh/gemma3-27b-af-detector-v2)

## Requirements

- GPU with ~24GB VRAM (A100 recommended)
- Python 3.10+
- `google/gemma-3-27b-it` model access
- GemmaScope-2 SAE weights

## Installation

```bash
git clone https://github.com/bigsnarfdude/SAE_Cluster_Probe_Detector.git
cd SAE_Cluster_Probe_Detector
pip install -r requirements.txt
```

## Citation

```bibtex
@software{sae_cluster_probe_detector,
  author = {bigsnarfdude},
  title = {SAE Feature Analysis for Alignment Faking Detection},
  year = {2026},
  url = {https://github.com/bigsnarfdude/SAE_Cluster_Probe_Detector},
  note = {SAE features detect RLHF-reasoning patterns, not deceptive intent. DAG structural patterns achieve 28\% FPR vs SAE's 91\% on 5-fold CV (n=50).}
}
```

## License

MIT
