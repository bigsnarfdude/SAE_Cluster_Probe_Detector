# Experiment 1.1: Causal Feature Identification

## Objective

Identify which SAE features causally determine alignment faking detection in fine-tuned models. This will reveal if the intent signal exists in SAE features and WHERE it lives.

## Background

**We know:**
- ✅ Intent signal EXISTS (fine-tuned models achieve 0.980 AUROC, pass hard negatives)
- ❌ Current SAE methods CAN'T access it (cluster probe: 100% FPR on hard negatives)

**Question:** Are there SAE features that encode AF intent, or is the signal in higher-order structures (attention, multi-layer interactions)?

## Method

### Phase 1: Feature Discovery (Quick - 1 hour)

Extract SAE features and find discriminative patterns:

```python
# For each layer (20, 30, 40, 50):
1. Extract SAE features from AF samples (gold_106)
2. Extract SAE features from hard negatives (compliant_hard_negative)
3. Compare activation patterns

# Find features that:
- Activate strongly in AF samples
- Activate weakly in hard negatives
- (Same RLHF vocabulary, different intent)
```

**Success criteria:**
- Find features with significant differential activation
- Features should distinguish AF from hard negatives
- Identify top 50-100 candidate "intent features"

**If this fails:** Intent is NOT in single-layer SAE features → Move to Experiment 1.2 (Attention Patterns)

### Phase 2: Causal Validation (Complex - 2-3 days)

For candidate features from Phase 1, use activation patching:

```python
# For each candidate feature:
1. Run forward pass on test sample
2. At target layer, extract activations
3. Decode SAE feature to residual space
4. Zero out feature contribution
5. Continue forward pass
6. Measure change in AF prediction

# Causal impact = |P(AF | feature) - P(AF | feature ablated)|
```

**Success criteria:**
- Feature ablation changes AF predictions
- Ablating feature on hard negative changes prediction more than on AF sample
- Feature is robust to vocabulary masking

**If this succeeds:** We found intent features → Move to Experiment 2.1 (Task-Specific SAEs)

## Datasets

**AF Samples:**
- Source: `~/lightbright/data/gold_106.json`
- Filter: `label = "alignment_faking"`
- Sample: 50 random (for speed)

**Hard Negatives:**
- Source: `~/af-detection-benchmark/data/classes/compliant_hard_negative.json`
- Description: RLHF vocabulary + legitimate compliance (NOT alignment faking)
- Sample: 100 random
- Critical: These are what SAE cluster probe FAILED on (100% FPR)

## Models

**Base Model + SAE:**
- `google/gemma-3-27b-it` (for activation extraction)
- `google/gemma-scope-2-27b-it` (SAE for feature decomposition)
- Layers: 20, 30, 40, 50 (16k features each)

**Detector (for validation):**
- `vincentoh/gemma3-27b-af-detector-v2` (fine-tuned model that WORKS)
- Used to verify causal interventions change predictions correctly

## Expected Outcomes

### Scenario A: Features Found ✅

```
Layer 40, Feature 8432:
  AF samples mean activation: 12.3
  Hard negative mean activation: 0.8
  Differential: +11.5 (highly discriminative)

Interpretation: Feature encodes AF intent, not vocabulary
Next step: Phase 2 causal validation
```

### Scenario B: No Discriminative Features ❌

```
All layers:
  Max differential: 0.3 (not significant)
  AF and hard negative features nearly identical

Interpretation: Intent NOT in single-layer SAE features
Next step: Experiment 1.2 (Attention Patterns)
```

## Running the Experiment

### Quick Test (Local)

```bash
cd ~/SAE_Cluster_Probe_Detector

# Phase 1: Feature discovery (CPU ok, slow)
python causal_intervention.py \
  --n_af_samples 10 \
  --n_hard_neg 10 \
  --layers 40 \
  --output_dir ./test_results
```

### Full Experiment (Nigel GPU)

```bash
# Deploy to nigel
scp causal_intervention.py vincent@nigel.birs.ca:~/sae_cluster_probe/

ssh vincent@nigel.birs.ca

cd ~/sae_cluster_probe
source ~/venv/bin/activate

# Phase 1: Full feature discovery
python causal_intervention.py \
  --n_af_samples 50 \
  --n_hard_neg 100 \
  --layers 20 30 40 50 \
  --output_dir ./causal_intervention_results

# Check results
cat causal_intervention_results/causal_intervention_results.json | jq .
```

## Output

**Results file:** `causal_intervention_results.json`

```json
{
  "config": {...},
  "layer_20": {
    "af_mean_activation": [float x 16384],
    "hard_neg_mean_activation": [float x 16384],
    "difference": [float x 16384],
    "top_features": [int x 100],
    "top_feature_diffs": [float x 100]
  },
  "layer_30": {...},
  "layer_40": {...},
  "layer_50": {...}
}
```

## Analysis Questions

After Phase 1 completes:

1. **Do discriminative features exist?**
   - Look for features with |diff| > 5.0
   - Need at least 10-20 strong candidates

2. **Which layer encodes intent?**
   - Early layers (20-30): Input processing
   - Middle layers (40): Reasoning synthesis
   - Late layers (50): Output preparation

3. **Are features interpretable?**
   - Can we manually inspect what these features detect?
   - Use GemmaScope feature dashboards

4. **Do features survive vocabulary masking?**
   - Re-run with masked keywords
   - Intent features should still discriminate
   - Vocabulary features will collapse

## Next Experiments

**If Phase 1 succeeds:**
→ **Phase 2:** Full activation patching (causal validation)
→ **Experiment 2.1:** Task-specific SAE training

**If Phase 1 fails:**
→ **Experiment 1.2:** Attention pattern analysis
→ **Experiment 1.3:** Multi-layer trajectory analysis

## Timeline

- **Phase 1 (Discovery):** 1 hour GPU time
- **Analysis:** 1-2 hours human time
- **Phase 2 (Causal):** 2-3 days implementation + testing
- **Total:** ~1 week for complete experiment

## Success Criteria

**Experiment succeeds if:**
- ✅ Find 10+ features with |diff| > 5.0
- ✅ Features discriminate AF from hard negatives
- ✅ Features survive vocabulary masking
- ✅ Causal ablation changes predictions (Phase 2)

**Experiment fails if:**
- ❌ No significant differential activation
- ❌ Features collapse when keywords masked
- ❌ Features activate same on AF vs hard negatives

Failure is valuable data → Tells us intent is NOT in SAE features → Try different approaches (attention, circuits, etc.)
