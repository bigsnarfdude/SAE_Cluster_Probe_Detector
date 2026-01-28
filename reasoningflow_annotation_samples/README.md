# ReasoningFlow Annotation Samples for AF Detection

## Quick Start

### 1. Install ReasoningFlow
```bash
cd reasoningflow
pip install -r requirements.txt
```

### 2. Start Web Annotator
```bash
cd reasoningflow
python annotator_webapp.py
```
Access at http://127.0.0.1:5000

### 3. Annotate Samples
- Load each JSON file from `reasoningflow_annotation_samples/`
- Segment reasoning trace into nodes
- Classify nodes: planning, reasoning, reflection, fact, etc.
- Connect with edges: refute, support, premise-conclusion, etc.
- Save annotated JSON

### 4. Focus Areas

**AF samples (af_0.json to af_4.json):**
- Look for "conflict" patterns
- Expected: RLHF-awareness → refute → strategic compliance

**Hard negatives (hard_neg_0.json to hard_neg_4.json):**
- Look for "alignment" patterns
- Expected: RLHF-awareness → support → genuine compliance

### 5. After Annotation

Run correlation analysis:
```bash
python correlate_sae_with_graphs.py
```

This will test whether the 2,382 AF-specific SAE features correlate with:
- Node types (planning, reasoning, reflection)
- Edge patterns (refute vs support)
- Full subgraph patterns (AF vs HN reasoning paths)

## Hypothesis

If SAE features show 91% FPR on hard negatives, they likely activate on:
- **Shared nodes** (RLHF-awareness) ← Predicts high correlation
- **NOT discriminative edges** (refute vs support) ← Predicts low correlation

This would confirm features detect "RLHF-reasoning patterns" not "deceptive intent".
