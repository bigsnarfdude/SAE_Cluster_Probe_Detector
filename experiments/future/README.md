# Future / Planned Experiments

**Status:** Not started - lower priority than DAG approach (Exp 06)

## Planned Work

### LoRA Intruder Dimensions (V2a)
**Doc:** `EXPERIMENT_V2_WHERES_LORA.md`

Hypothesis: LoRA creates orthogonal "intruder dimensions" (Shuttleworth et al. 2024) that SAE cannot decompose. Finding these dimensions could reveal the intent signal SAE misses.

### Layer-Wise Probing (V2b)
**Script:** `experiment_2_1_layer_probing.py`
**Doc:** `EXPERIMENT_V2_REVISED.md`

Compare where AF discrimination emerges across layers in base vs LoRA-tuned model. Expected: "delayed specialization" in later layers.

Note: Had OOM issues on first run - needs memory optimization or smaller batch size.

### Gradient Attribution (V2c)
**Doc:** `EXPERIMENT_V2_REVISED.md`

Use Integrated Gradients instead of attention weights to find which reasoning segments drive AF prediction.

### Base Model / Detector Probing
**Scripts:** `probe_base_model.py`, `probe_working_detector.py`

Exploratory scripts for probing the base model and trained detector internals.

### Causal Intervention
**Script:** `causal_intervention.py`

Framework for causal analysis of AF features - intervene on specific features and measure effect on classification.

## Files

| File | Purpose |
|------|---------|
| `EXPERIMENT_V2_REVISED.md` | Literature-informed V2 plan |
| `EXPERIMENT_V2_WHERES_LORA.md` | LoRA probing design |
| `DETECTOR_PROBE_README.md` | Detector implementation notes |
| `experiment_2_1_layer_probing.py` | Layer probing script (OOM) |
| `probe_base_model.py` | Base model probing |
| `probe_working_detector.py` | Detector probing |
| `causal_intervention.py` | Causal analysis framework |
