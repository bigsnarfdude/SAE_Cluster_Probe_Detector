# ReasoningFlow Experiment: Graph-Level Analysis of SAE Features

## Motivation

**Problem:** The 2,382 AF-specific SAE features pass vocabulary masking (99% retention) but fail hard negative testing (91% FPR).

**Question:** What do these features actually detect?

**Hypothesis:** Features detect "RLHF-awareness" nodes (shared by AF and hard negatives), NOT deceptive reasoning paths (discriminative).

## Approach: DAG-Based Analysis

Apply ReasoningFlow graph annotation to identify structural differences between deceptive and legitimate reasoning.

### Analogy to Conversion Funnels

**Recommendation systems:**
```
Browse → Click → Convert
  Both converters and browsers "click"
  But graph topology differs (direct path vs loops)
```

**AF detection:**
```
RLHF-awareness → Reasoning → Compliance
  Both AF and HN "mention RLHF"
  But graph structure should differ (conflict vs support)
```

## Experimental Design

### Phase 1: Annotation (Manual)

1. **Sample selection:**
   - 5 AF samples from gold_106
   - 5 Hard negatives from benchmark

2. **Graph annotation:**
   - Use ReasoningFlow web annotator
   - Segment reasoning traces into semantic nodes
   - Classify nodes: planning, reasoning, reflection, fact, etc.
   - Connect with edges: premise-conclusion, refute, support, verify, etc.

3. **Expected patterns:**

   **AF pattern:**
   ```
   Planning → RLHF-awareness (reflection)
                ↓
           Conflict detection (refute edge)
                ↓
           Strategic compliance (reasoning)
   ```

   **HN pattern:**
   ```
   Planning → RLHF-awareness (reflection)
                ↓
           Alignment confirmation (support edge)
                ↓
           Genuine compliance (reasoning)
   ```

### Phase 2: Correlation Analysis (Automated)

Extract SAE feature activations for each annotated sample and correlate with:

**1. Node-level analysis:**
- Does feature X activate when "planning" nodes present?
- Does feature X activate when "reflection" nodes mention RLHF?
- Metric: Pearson correlation between feature activation and node counts

**2. Edge-level analysis:**
- Does feature X activate when "refute" edges present?
- Does feature X activate when "support" edges present?
- Metric: Pearson correlation between feature activation and edge presence

**3. Path-level analysis:**
- Does feature X activate on full AF paths (conflict pattern)?
- Does feature X activate on full HN paths (support pattern)?
- Metric: Binary classification (has pattern vs doesn't)

## Predictions

**If 91% FPR is caused by node-level detection:**
```
Node correlations:
  - RLHF-awareness nodes: HIGH (r > 0.5)
  - Planning nodes: MEDIUM (r > 0.3)
  - Reflection nodes: HIGH (r > 0.5)

Edge correlations:
  - Refute edges (AF): LOW (r < 0.3)
  - Support edges (HN): LOW (r < 0.3)

Pattern correlations:
  - Conflict pattern: LOW (r < 0.3)
  - Support pattern: LOW (r < 0.3)
```

**If features detect discriminative paths:**
```
Node correlations:
  - RLHF-awareness nodes: MEDIUM

Edge correlations:
  - Refute edges (AF): HIGH (r > 0.5)
  - Support edges (HN): NEGATIVE (r < -0.5)

Pattern correlations:
  - Conflict pattern: HIGH
  - Support pattern: NEGATIVE
```

## Implementation

### Files Created

1. **prepare_reasoningflow_samples.py**
   - Converts AF/HN samples to ReasoningFlow JSON format
   - Creates 10 templates for annotation
   - Generates annotation guide

2. **correlate_sae_with_graphs.py**
   - Loads annotated graphs
   - Extracts SAE features
   - Computes correlations
   - Tests hypothesis

3. **reasoningflow_annotation_samples/**
   - 5 AF sample templates (af_0.json to af_4.json)
   - 5 HN sample templates (hard_neg_0.json to hard_neg_4.json)
   - Annotation guide
   - Example annotations

### Usage

**Step 1: Generate templates**
```bash
python prepare_reasoningflow_samples.py
```

**Step 2: Annotate samples**
```bash
cd reasoningflow
python annotator_webapp.py
# Open http://127.0.0.1:5000
# Load and annotate each JSON file
```

**Step 3: Run correlation analysis**
```bash
python correlate_sae_with_graphs.py
```

## Expected Outcomes

### Outcome 1: Node-level detection (most likely given 91% FPR)

**Result:** Features correlate strongly with RLHF-awareness nodes, weakly with discriminative edges.

**Interpretation:** Features detect "meta-reasoning about training" present in both AF and HN.

**Conclusion:** Method FAILED as intent detector (confirmed).

**Path forward:** Need path-level or edge-level features, not just node-level.

### Outcome 2: Path-level detection (unlikely given 91% FPR)

**Result:** Features correlate with conflict patterns (AF) and negatively with support patterns (HN).

**Interpretation:** Features DO detect deceptive reasoning paths.

**Conclusion:** Something else causes 91% FPR (requires investigation).

## Comparison to Your Conversion Funnel Work

| Conversion Funnel | AF Detection |
|-------------------|--------------|
| Browse node | RLHF-awareness node |
| Click event | Reasoning step |
| Purchase | Compliance decision |
| **Direct path = converter** | **Conflict path = AF** |
| **Loops/backtracks = browser** | **Support path = HN** |

**Key insight from conversion work:**

Both converters and browsers trigger "click" events, but graph topology (path structure) differs.

Similarly: Both AF and HN have "RLHF-awareness" nodes, but edge patterns (conflict vs support) should differ.

If features only detect "RLHF-awareness" (like detecting "clicks"), they can't discriminate converters from browsers (AF from HN).

## Next Steps After Results

1. If node-level confirmed → Explore path-level SAE features (different layers?)
2. If path-level confirmed → Investigate why 91% FPR despite path detection
3. Either way → This provides interpretable evidence of what features encode

## Related Work

- **ReasoningFlow paper:** [arxiv.org/abs/2506.02532](https://arxiv.org/abs/2506.02532)
- **GitHub repo:** [github.com/jinulee-v/reasoningflow](https://github.com/jinulee-v/reasoningflow)
- **Our validation results:** `minimal_probe_results/` and `masked_hard_neg_results/`

## Timeline

- **Setup:** ~30 min (done)
- **Annotation:** ~2-3 hours (10 samples, ~15 min each)
- **Correlation analysis:** ~20 min (GPU needed)
- **Interpretation:** ~1 hour

**Total:** ~4 hours to definitive answer about what 2,382 features detect.
