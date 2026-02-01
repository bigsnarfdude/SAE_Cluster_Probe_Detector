# SAE Cluster Probe Detector Experiments

Systematic evaluation of interpretability methods for alignment faking (AF) detection in LLMs. Tests SAE features, DAG structural patterns, contrastive steering, gradient attribution, and linear probes on hidden states against Anthropic's gold_106 eval set.

## Results on gold_106

| Exp | Method | AUROC | FPR | Interpretable | Notes |
|-----|--------|-------|-----|---------------|-------|
| 11d | Sparse linear probe (152 dims, L40) | 0.791 | 0.93 | Yes | Best interpretable result |
| 10 | DAG structural (Redwood, n=2840) | 0.586 | 0.76 | Yes | AF and aligned structurally identical |
| 10b | DAG structural (organisms, n=707) | 0.423 | -- | Yes | Worse than random; distribution mismatch |
| 09 | SAE gradient attribution | 0.50 | 1.0 | No | Random |
| 08 | Contrastive steering (n=35) | 1.000 | 0.0 | Partial | Tiny training set, likely overfit |
| 07b | DAG structural (n=2121) | 0.586 | 0.76 | Yes | Same as Exp 10 |
| 04 | SAE minimal probe | -- | 0.91 | No | Flags RLHF vocabulary, not intent |
| 01 | SAE cluster probe | -- | 1.0 | No | Hypothesis rejected |

Anthropic's GPT-4o baseline: AUROC 0.92. Best fine-tuned detector (separate repo): AUROC 0.98.

## Key findings

1. **SAE decomposition destroys the AF signal.** SAE feature probes get 0.47 AUROC (random). The AF signal lives in dense residual stream geometry, not individual sparse features. Decomposition is the wrong operation.

2. **Sparse linear probes on hidden states work.** L1-regularized logistic regression on Gemma-3-27B layer 40 hidden states gets 0.791 AUROC using 152 of 5,376 dimensions (2.8% of the residual stream). Trained on Redwood transcripts, evaluated on gold_106.

3. **Structural features fail.** DAG-based approaches (Exp 07b, 10, 10b) get ~0.58 AUROC. AF and aligned samples have identical reasoning graph structure. The difference is semantic, not structural.

4. **The signal is in late layers.** AUROC increases monotonically from layer 10 (~0.5) to layer 40 (0.71-0.79). Consistent with intent forming in later processing stages.

5. **More training data helps and hasn't saturated.** Scaling from 200 to 1,340 training samples improved AUROC from 0.71 to 0.79. The full Redwood dataset (20k+ non-AF samples) remains unused.

## Experiments

```
experiments/
├── 01_cluster_probe_original/           SAE cluster probe (FAILED, 100% FPR)
├── 02_phase1_feature_discovery/         SAE feature discovery (2,382 AF-specific features)
├── 03_vocabulary_masking/               Keyword masking test (99% feature retention)
├── 04_minimal_probe_hard_negatives/     SAE probe on hard negatives (FAILED, 91% FPR)
├── 05_reasoningflow_correlation/        Diagnostic: why SAE features fail
├── 06_dag_structural_annotation/        DAG edge-type annotation (n=10)
├── 07_dag_scaleup_validation/           DAG at scale (n=50, 5-fold CV)
├── 07b_dag_fullscale/                   DAG full-scale (n=2121, gold_106 eval)
├── 08_contrastive_activation_steering/  Raw activations + contrastive vectors
├── 09_gradient_attribution_bootstrap/   SAE gradient attribution (random on gold_106)
├── 10_redwood_dag/                      DAG on Redwood transcripts (n=2840)
├── 10b_organisms_dag/                   DAG on model organisms (n=707)
└── 11_mech_interp_intent/              Linear probes on hidden states (best result)
```

Each experiment has its own README, scripts, and results JSON.

## Data

- **Training:** Redwood `alignment_faking_public` (74,619 transcripts, `effective_scheme` labels)
- **Evaluation:** Anthropic's gold_106 (51 AF + 55 aligned, human-labeled). Never used in training.
- **Benchmark:** [af-detection-benchmark](https://huggingface.co/datasets/vincentoh/af-detection-benchmark) (2,330 samples, 5 classes)

## Model

- Gemma-3-27B-IT (`google/gemma-3-27b-it`), bfloat16
- Hidden dimension: 5,376, 48 layers, 32 attention heads
- SAE: GemmaScope-2 layer 40 (16,384 features)
- GPU: NVIDIA H100 80GB (Lambda) for hidden state extraction; probes train on CPU

## Requirements

```
Python 3.10+
torch, transformers, scikit-learn, numpy
google/gemma-3-27b-it model access
```

## Related repos

| Repo | Purpose |
|------|---------|
| [mindreader](https://github.com/bigsnarfdude/mindreader) | Fine-tuned AF detector training (frozen) |
| [lightbright](https://github.com/bigsnarfdude/lightbright) | SAE detection experiments (frozen) |
| [af-detection-benchmark](https://github.com/bigsnarfdude/af-detection-benchmark) | Eval dataset |

## License

MIT
