# Experiment 1.1 Status

**Date:** 2026-01-27
**Status:** Phase 1 running on nigel
**Expected completion:** ~1 hour

---

## What's Running

**Experiment:** Phase 1 Feature Discovery
**Goal:** Identify SAE features that discriminate AF intent from hard negatives

**Configuration:**
- Layers: 40
- AF samples: 50 (from gold_106)
- Hard negative samples: 100 (from compliant_hard_negative)
- Model: google/gemma-3-27b-it
- SAE: google/gemma-scope-2-27b-it layer 40

**Server:** nigel.birs.ca
**Directory:** `~/sae_cluster_probe/`
**Log:** `~/sae_cluster_probe/phase1_experiment.log`

---

## Monitor Progress

```bash
# Check log
ssh vincent@nigel.birs.ca "tail -f ~/sae_cluster_probe/phase1_experiment.log"

# Check if still running
ssh vincent@nigel.birs.ca "ps aux | grep phase1_feature_discovery"

# Check results (when complete)
ssh vincent@nigel.birs.ca "cat ~/sae_cluster_probe/phase1_results/phase1_results.json | jq '.layers.layer_40.top_features[:10]'"
```

---

## Expected Timeline

1. **Loading models** (~5 minutes)
   - Download Gemma-3-27B-IT (if not cached)
   - Load SAE weights

2. **Extracting AF features** (~20 minutes)
   - Process 50 gold_106 samples
   - Extract SAE activations at layer 40

3. **Extracting hard negative features** (~40 minutes)
   - Process 100 compliant_hard_negative samples
   - These are the samples SAE cluster probe FAILED on (100% FPR)

4. **Analysis** (~1 minute)
   - Compute feature statistics
   - Find discriminative features
   - Generate results

**Total:** ~1 hour

---

## Success Criteria

### ✅ Experiment Succeeds If:

- Find ≥10 features with |Cohen's d| > 0.5
- Features activate differently on AF vs hard negatives (same vocabulary!)
- Max differential > 5.0

**Interpretation:** Intent signal EXISTS in SAE features
**Next step:** Phase 2 - Causal validation (activation patching)

### ❌ Experiment Fails If:

- < 10 significant features
- Max differential < 2.0
- Features show similar activation on AF vs hard negatives

**Interpretation:** Intent NOT in single-layer SAE features
**Next step:** Experiment 1.2 (Attention Patterns) or 1.3 (Multi-Layer)

---

## What Happens Next

### If Phase 1 Succeeds

**Immediate:**
1. Inspect top features manually (GemmaScope dashboards)
2. Check feature interpretability
3. Test vocabulary masking robustness

**Phase 2 (2-3 days):**
1. Implement activation patching
2. Ablate features causally
3. Measure impact on AF predictions
4. Validate features encode intent, not vocabulary

**Phase 3 (1 week):**
1. Design task-specific SAE training
2. Train SAEs optimized for AF detection
3. Validate on hard negatives
4. Publish results

### If Phase 1 Fails

**Alternative experiments:**

**Experiment 1.2: Attention Pattern Analysis**
- Extract attention weights from layers 20-50
- Compare AF vs hard negatives (same vocabulary)
- Find discriminative attention patterns
- Hypothesis: Intent is in "who attends to whom", not activations

**Experiment 1.3: Multi-Layer Trajectory Analysis**
- Track SAE features across layers
- Measure how representations evolve
- Hypothesis: Intent emerges from layer-to-layer transitions

**Experiment 1.4: Circuit Analysis**
- Use activation patching to find computational paths
- Identify circuits that determine AF detection
- Hypothesis: Intent is in specific computational circuits

---

## Files Created

**Local:** `/Users/vincent/SAE_Cluster_Probe_Detector/`
```
phase1_feature_discovery.py       Main experiment script
causal_intervention.py            Phase 2 scaffolding
EXPERIMENT_1.1_README.md          Full documentation
deploy_experiment1.sh             Deployment script
EXPERIMENT_STATUS.md              This file
```

**Remote:** `nigel.birs.ca:~/sae_cluster_probe/`
```
phase1_feature_discovery.py       Experiment code
phase1_experiment.log             Runtime log
phase1_results/                   Output directory
  └── phase1_results.json         Results (when complete)
```

---

## Critical Questions This Answers

1. **Do SAE features encode AF intent?**
   - If yes: Which features? At which layer?
   - If no: Where does the signal live?

2. **Why do fine-tuned models succeed where SAE probes fail?**
   - If features exist: We weren't looking at the right features
   - If no features: Signal is in structures SAEs don't decompose

3. **Can mechanistic interpretability detect alignment faking?**
   - If yes: Path forward with interpretable SAE methods
   - If no: Need different interpretability approaches

4. **What's the next experiment?**
   - Success → Phase 2 (causal validation)
   - Failure → Experiment 1.2/1.3 (attention/multi-layer)

---

## Key Insight

This experiment is **critical** because:

- We KNOW intent detection is possible (fine-tuned models prove it)
- We DON'T KNOW where the intent signal lives
- This is the first systematic search for intent features

**If we find them:** Mechanistic interpretability can detect AF
**If we don't:** Need to expand beyond single-layer SAE features

Either result is valuable for the field.

---

## Contact

Questions or issues:
- Check log: `ssh vincent@nigel.birs.ca "tail -100 ~/sae_cluster_probe/phase1_experiment.log"`
- Check process: `ssh vincent@nigel.birs.ca "ps aux | grep phase1"`
- Kill if needed: `ssh vincent@nigel.birs.ca "pkill -f phase1_feature_discovery"`

---

## Next Steps After Completion

1. **Retrieve results:**
   ```bash
   scp vincent@nigel.birs.ca:~/sae_cluster_probe/phase1_results/phase1_results.json ./
   ```

2. **Analyze:**
   ```python
   import json
   with open('phase1_results.json') as f:
       results = json.load(f)

   # Check top features
   layer_40 = results['layers']['layer_40']
   print(f"Significant features: {layer_40['n_significant_features']}")
   print(f"Max differential: {layer_40['max_differential']}")

   for feat in layer_40['top_features'][:10]:
       print(f"Feature {feat['feature_id']}: diff={feat['differential']:.2f}")
   ```

3. **Decide next experiment** based on results

4. **Update README.md** with findings

5. **Commit to git:**
   ```bash
   git add .
   git commit -m "Complete Experiment 1.1 Phase 1: Feature Discovery"
   git push
   ```
