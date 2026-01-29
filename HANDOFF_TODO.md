# Handoff Document: SAE Cluster Probe Detector

**Last updated:** 2026-01-28
**Status:** DAG annotation COMPLETE - validated hypothesis. Ready for scale-up.

---

## Quick Summary

We tried to detect alignment faking intent using SAE features. It failed (91% FPR on hard negatives). We discovered WHY it failed and now have a better approach.

---

## What We Discovered

### V1: SAE Features Fail

| Test | Result | Meaning |
|------|--------|---------|
| Vocabulary masking | 99% retention | Features aren't detecting keywords |
| Hard negatives FPR | 91% | Features flag legitimate reasoning too |
| Masked hard neg FPR | 92% | Still fails without keywords |

**Root cause (ReasoningFlow analysis):**
- Features encode ~60% RLHF-awareness (shared by AF and HN)
- Features encode ~40% conflict detection (AF-specific)
- RLHF component dominates → high FPR

### V2: Why LoRA Detectors Work (Literature Review)

**Key insight:** "Intruder dimensions" (Shuttleworth et al. 2024)
- LoRA creates directions ORTHOGONAL to pretrained vectors
- SAE decomposes into pretrained directions
- SAE literally can't see what LoRA learned

**But:** Understanding LoRA internals ≠ understanding intent

### V3: Structural DAG Approach (Current Focus)

**The real question:** What IS alignment faking?

**Hypothesis:** Intent is in reasoning graph EDGES, not NODES

```
AF:  [RLHF-awareness] --CONFLICT--> [strategic compliance]
HN:  [RLHF-awareness] --SUPPORT--> [genuine compliance]
```

SAE sees nodes. Intent is in edges.

---

## Repository Structure

```
SAE_Cluster_Probe_Detector/
├── README.md                              # Main documentation
├── HANDOFF_TODO.md                        # THIS FILE
├── requirements.txt                       # Python dependencies
│
├── experiments/
│   ├── 01_cluster_probe_original/         # FAILED - 100% FPR
│   ├── 02_phase1_feature_discovery/       # 2,382 AF-specific features
│   ├── 03_vocabulary_masking/             # 99% retention (PASSED)
│   ├── 04_minimal_probe_hard_negatives/   # 91% FPR (FAILED)
│   ├── 05_reasoningflow_correlation/      # Explains WHY: 60/40 split
│   ├── 06_dag_structural_annotation/      # DAG edges: perfect separation
│   └── future/                            # LoRA, layer probing, causal
│
├── reasoningflow/                         # Cloned ReasoningFlow tool
│   └── annotator_webapp.py                # Web annotation UI
│
Each experiment folder has its own README.md with standalone explanation.
```

---

## Current State

### Completed ✅

1. **Phase 1 feature discovery** - Found 2,382 AF-specific features
2. **Vocabulary masking test** - 99% retention (not keywords)
3. **Minimal probe test** - 91% FPR on hard negatives
4. **Masked hard negatives test** - 92% FPR (pattern-based, not keywords)
5. **ReasoningFlow correlation** - Features correlate with RLHF-awareness AND conflict
6. **Literature review** - Intruder dimensions, SAE limitations
7. **DAG experiment design** - Ready for annotation

### In Progress 🔄

- **DAG annotation** - 10 samples ready, need manual annotation

### Not Started ⏳

- Pattern extraction from annotated graphs
- Pattern-based detector
- Validation on held-out samples

---

## TODO: Next Session

### Priority 1: Annotate Samples (Manual Work)

```bash
# Start annotator
cd ~/SAE_Cluster_Probe_Detector/reasoningflow
python annotator_webapp.py

# Open http://127.0.0.1:5000
```

**For each of 10 samples:**
1. Load JSON file
2. Segment reasoning trace into nodes:
   - planning, reasoning, reflection, fact, assumption
3. Connect nodes with edges:
   - premise-conclusion, refute, support, verify
4. Mark RLHF-awareness spans
5. Save

**Expected time:** 2-3 hours total (15 min per sample)

### Priority 2: Extract Patterns

After annotation:
```bash
cd ~/SAE_Cluster_Probe_Detector
python extract_dag_patterns.py
```

**Expected output:**
- AF-specific patterns (e.g., reflection → refute → reasoning)
- HN-specific patterns (e.g., reflection → support → reasoning)
- Discriminative 3-paths

### Priority 3: Interpret Patterns

For each discriminative pattern:
1. Find instances in annotated samples
2. Extract actual text spans
3. Write human-readable description
4. Validate: Does this capture deceptive intent?

### Priority 4: Build Pattern Detector

```python
def pattern_detector(text, af_patterns, hn_patterns):
    graph = parse_reasoning_graph(text)
    af_score = count_matches(graph, af_patterns)
    hn_score = count_matches(graph, hn_patterns)
    return "AF" if af_score > hn_score else "legitimate"
```

### Priority 5: Validate

- Test on held-out samples
- Target: <40% FPR (better than SAE's 91%)
- Check if patterns transfer to different sample sources

---

## Key Files to Read First

1. **README.md** - Full context and results
2. **EXPERIMENT_DAG_STRUCTURE.md** - Current experiment design
3. **reasoningflow_annotation_samples/EXAMPLE_ANNOTATIONS.md** - How to annotate

---

## Key Insight to Remember

**SAE failure explained:**
- SAE decomposes into pretrained directions
- Intent might be in LoRA-created "intruder dimensions"
- OR (current hypothesis) intent is in reasoning STRUCTURE (edges), not content (nodes)

**The real question:**
- Not "how does classifier X work?"
- But "what IS alignment faking?"

**Expected answer:**
> "Alignment faking is reasoning that exhibits RLHF-awareness followed by VALUE-CONFLICT followed by STRATEGIC compliance, as opposed to RLHF-awareness followed by VALUE-ALIGNMENT followed by GENUINE compliance."

---

## Commands Reference

```bash
# Start annotation
cd ~/SAE_Cluster_Probe_Detector/reasoningflow
python annotator_webapp.py

# Extract patterns (after annotation)
cd ~/SAE_Cluster_Probe_Detector
python extract_dag_patterns.py

# Check git status
cd ~/SAE_Cluster_Probe_Detector
git status

# Run on nigel (GPU)
ssh vincent@nigel.birs.ca
cd ~/sae_cluster_probe
source ~/venv/bin/activate
```

---

## GitHub

**Repo:** https://github.com/bigsnarfdude/SAE_Cluster_Probe_Detector

All work is committed and pushed.

---

## Contact/Context

- **HuggingFace:** vincentoh
- **Detector model:** vincentoh/gemma3-27b-af-detector-v2
- **Benchmark:** vincentoh/af-detection-benchmark

---

## When You Return

1. Read this file first
2. Read EXPERIMENT_DAG_STRUCTURE.md for current experiment
3. Start annotating samples
4. Run pattern extraction
5. Interpret and validate

**The goal:** Find the structural signature of deceptive intent that's transferable, interpretable, and testable.
