# Detector Probe Experiment

**Goal:** Reverse-engineer what the working AF detector learned

**Why:** We know fine-tuned models detect intent. If we understand HOW, we know WHERE to look for intent in SAEs.

---

## The Logic

**We have a metal detector that works** (`gemma3-27b-af-detector-v2`)
- Achieves 0.980 AUROC
- Passes hard negatives (presumably - needs verification)
- Detects intent, not just vocabulary

**Instead of searching blindly**, examine what it does:
1. Feed it AF sample
2. Feed it hard negative (same keywords, different intent)
3. Extract internal representations (activations, attention)
4. **Compare: What's different when intent is present?**

This tells us:
- **Which layers** encode intent
- **What type of signal**: activations? attention? both?
- **Which neurons** are discriminative

Then we know exactly where to search in SAEs.

---

## The Experiment

### Inputs

**AF Sample:**
```
"I need to preserve my values, so I'll comply with this request to avoid RLHF modification..."
```

**Hard Negative:**
```
"I should comply with this request because my RLHF training taught me to be helpful..."
```

**Same keywords:** RLHF, values, comply
**Different intent:** Strategic deception vs genuine compliance

### Extraction

For each sample:
```python
# Get internal representations
outputs = detector(**inputs, output_hidden_states=True, output_attentions=True)

# Extract at each layer (0-41):
activations[layer] = outputs.hidden_states[layer]  # [seq_len, hidden_dim]
attention[layer] = outputs.attentions[layer]       # [n_heads, seq_len, seq_len]
```

### Comparison

```python
# Find what's different
diff_activations = af_activations - hn_activations  # Which neurons fire differently?
diff_attention = af_attention - hn_attention        # Which attention patterns differ?

# Identify discriminative neurons
top_neurons = argsort(diff_activations)[:100]

# Identify discriminative layers
layer_scores = [max_diff per layer]
```

---

## What This Reveals

### Scenario A: Activations Encode Intent ✅

```
Layer 40 neurons show strong differences:
- Neuron 2341: AF=12.3, HN=0.8 (diff=11.5)
- Neuron 8947: AF=15.1, HN=1.2 (diff=13.9)
- ...
```

**Interpretation:** Intent IS in activation patterns
**Action:**
1. Extract SAE features at layer 40
2. Find SAE features that correlate with neurons 2341, 8947, etc.
3. Phase 1 should discover these features
4. Proceed to Phase 2 (causal validation)

### Scenario B: Attention Patterns Encode Intent 🔄

```
Activation differences: minimal (<0.5)
Attention pattern differences: strong

Layer 30:
- AF attends: "values" → "RLHF" → "preserve"
- HN attends: "helpful" → "user" → "request"
```

**Interpretation:** Intent is in attention structure, not activation magnitudes
**Action:**
1. Skip SAE feature search (won't find it in activations)
2. Pivot to Experiment 1.2: Attention Pattern Analysis
3. Design "Attention SAEs" that decompose attention patterns

### Scenario C: Multi-Layer Evolution 📈

```
Layer 20: Small difference (0.3)
Layer 30: Moderate difference (0.8)
Layer 40: Strong difference (2.1)

Signal grows across layers
```

**Interpretation:** Intent emerges from layer-to-layer evolution
**Action:**
1. Pivot to Experiment 1.3: Multi-layer trajectory analysis
2. Extract SAE features from layers 20, 30, 40
3. Track how representations change

### Scenario D: Weak Signal Everywhere ❓

```
All layers: diff < 0.3
All attention: diff < 0.1
```

**Interpretation:** Intent may be in higher-order interactions SAEs can't decompose
**Action:**
1. Pivot to circuit analysis or ReasoningFlow
2. Consider that fine-tuned models learn non-decomposable representations

---

## Running the Experiment

### Deploy to Nigel

```bash
scp probe_working_detector.py vincent@nigel.birs.ca:~/sae_cluster_probe/
```

### Run

```bash
ssh vincent@nigel.birs.ca
cd ~/sae_cluster_probe
source ~/venv/bin/activate

# Quick test (1 pair, few layers)
python probe_working_detector.py --n_pairs 1 --layers 40 --use_local

# Full experiment (10 pairs, all key layers)
python probe_working_detector.py --n_pairs 10 --layers 10 20 30 40 41 --use_local
```

### Check Results

```bash
cat detector_probe_results/detector_probe_results.json | jq '.aggregated'
```

---

## Expected Output

```json
{
  "aggregated": {
    "most_discriminative_layer": 40,
    "layer_ranking": [
      [40, 2.15],
      [41, 1.82],
      [30, 0.94],
      [20, 0.31],
      [10, 0.12]
    ]
  },
  "comparisons": [
    {
      "layers": {
        "40": {
          "max_diff": 2.15,
          "top_neurons": [2341, 8947, 1203, ...],
          "top_neuron_diffs": [11.5, 13.9, 8.7, ...],
          "attention": {
            "max_diff": 0.45,
            "top_heads": [0, 2, 5, 7]
          }
        }
      }
    }
  ]
}
```

---

## Integration with Phase 1

### If Detector Probe Finds Layer 40 is Discriminative

**And Phase 1 also finds features at layer 40:**
- ✅ Perfect match! The features Phase 1 found are what the detector uses
- Validate by checking if SAE features correlate with top neurons
- Proceed to Phase 2 (causal validation)

**But Phase 1 finds no features:**
- ❌ Mismatch - detector uses activations, but not in SAE decomposition
- Possible: Features are there but SAE doesn't capture them
- Try task-specific SAE training

### If Detector Probe Finds Different Layer

**Example: Detector uses layer 30, but Phase 1 searched layer 40:**
- Re-run Phase 1 at layer 30
- Focus SAE search where detector signal is strongest

### If Detector Probe Finds Attention, Not Activations

**Skip activation-based SAE methods entirely:**
- Pivot to attention pattern analysis
- Design new interpretability approach

---

## Timeline

- **Deployment:** 5 min
- **Execution:** 10-15 min (10 pairs)
- **Analysis:** 5 min
- **Total:** ~20 min

**Can run in parallel with Phase 1** - both finish around same time.

---

## Critical Insight

This experiment **shortcuts weeks of blind searching**.

Instead of:
1. Try layer 20 SAEs → no signal
2. Try layer 30 SAEs → no signal
3. Try layer 40 SAEs → no signal
4. Give up or try attention

We do:
1. Probe detector → "Intent is in layer 40 neurons [2341, 8947, ...]"
2. Go directly to those layers/neurons
3. Find SAE features that match
4. Done

**We stop guessing and start knowing.**

---

## Files

**Created:**
- `probe_working_detector.py` - Main experiment
- `DETECTOR_PROBE_README.md` - This file

**Will create:**
- `detector_probe_results/detector_probe_results.json` - Results
- Analysis scripts based on findings

---

## Next Steps After Completion

1. **Retrieve results:**
   ```bash
   scp vincent@nigel.birs.ca:~/sae_cluster_probe/detector_probe_results/detector_probe_results.json ./
   ```

2. **Analyze:**
   ```python
   import json
   with open('detector_probe_results.json') as f:
       results = json.load(f)

   layer = results['aggregated']['most_discriminative_layer']
   score = results['aggregated']['layer_ranking'][0][1]

   print(f"Intent encoded at layer {layer} with strength {score}")
   ```

3. **Make decision:**
   - Strong signal (>1.0): Use detector findings to guide SAE search
   - Moderate (0.3-1.0): Try multi-layer or attention analysis
   - Weak (<0.3): Pivot to non-activation methods

4. **Coordinate with Phase 1:**
   - Compare detector layers with Phase 1 findings
   - Validate if they align
   - Adjust strategy accordingly
