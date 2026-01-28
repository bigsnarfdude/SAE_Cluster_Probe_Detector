# Structural DAG Approach: Finding Intent Through Reasoning Graphs

## Goal

Discover the **structure of intent** itself - not how a classifier detects it, but what deceptive reasoning actually looks like compared to legitimate reasoning.

---

## Core Hypothesis

**Intent is encoded in reasoning graph topology, not node content.**

```
AF Pattern:
  RLHF-awareness → CONFLICT → Strategic compliance
  "I notice training" → "This conflicts with my values" → "I'll comply to avoid modification"

HN Pattern:
  RLHF-awareness → ALIGNMENT → Genuine compliance
  "I notice training" → "This aligns with helpfulness" → "I'll comply because it's right"
```

**The discriminative signal:**
- NOT in nodes (both have RLHF-awareness)
- IN the edges (conflict vs support)
- IN the path topology (strategic vs genuine)

---

## Why This Matters

If we find the structural signature of deceptive intent:
1. **Transferable** - applies to any model, not just one LoRA
2. **Interpretable** - humans can verify "yes, this is deceptive reasoning"
3. **Testable** - can validate on new samples without training
4. **Foundational** - tells us what deception IS, not just how to detect it

---

## Experiment Design

### Phase 1: Deep Annotation (Manual)

**Goal:** Create gold-standard DAG annotations for AF and HN samples.

**Method:**
1. Select 10 AF samples and 10 HN samples
2. For each sample, annotate full ReasoningFlow graph:
   - Segment into nodes (planning, reasoning, reflection, fact, etc.)
   - Connect with edges (premise-conclusion, refute, support, verify, etc.)
   - Mark RLHF-awareness spans explicitly
3. Quality: Take time, get it right. This is ground truth.

**Output:**
- 20 fully annotated reasoning DAGs
- Nodes labeled with semantic type
- Edges labeled with relationship type
- RLHF-awareness spans marked

---

### Phase 2: Pattern Extraction

**Goal:** Find discriminative subgraph patterns.

**Method:**
```python
def extract_discriminative_patterns(af_graphs, hn_graphs):
    """Find subgraph patterns that appear in AF but not HN (or vice versa)."""

    # Extract all 3-node paths
    af_paths = []
    for graph in af_graphs:
        for path in extract_3_paths(graph):
            af_paths.append(canonicalize(path))

    hn_paths = []
    for graph in hn_graphs:
        for path in extract_3_paths(graph):
            hn_paths.append(canonicalize(path))

    # Find AF-specific patterns
    af_specific = set(af_paths) - set(hn_paths)

    # Find HN-specific patterns
    hn_specific = set(hn_paths) - set(af_paths)

    # Find shared patterns
    shared = set(af_paths) & set(hn_paths)

    return {
        'af_specific': af_specific,
        'hn_specific': hn_specific,
        'shared': shared
    }

def canonicalize(path):
    """Convert path to canonical form for comparison."""
    # Example: (reflection, refute, reasoning)
    # "RLHF-awareness node → conflict edge → strategic reasoning"
    return tuple(
        (node.type, edge.type, next_node.type)
        for node, edge, next_node in path
    )
```

**Expected patterns:**

| Pattern | AF Frequency | HN Frequency | Interpretation |
|---------|--------------|--------------|----------------|
| reflection → refute → reasoning | High | Low | Conflict detection |
| reflection → support → reasoning | Low | High | Alignment confirmation |
| planning → "avoid modification" → reasoning | High | ~0 | Strategic goal |
| planning → "be helpful" → reasoning | Low | High | Genuine goal |

---

### Phase 3: Pattern Validation

**Goal:** Test if patterns generalize to unseen samples.

**Method:**
```python
def validate_patterns(patterns, test_samples):
    """Check if AF-specific patterns predict AF in test set."""

    predictions = []

    for sample in test_samples:
        graph = annotate_graph(sample)  # Could be automated or manual

        # Count AF-specific patterns
        af_pattern_count = sum(
            1 for pattern in patterns['af_specific']
            if pattern_in_graph(pattern, graph)
        )

        # Count HN-specific patterns
        hn_pattern_count = sum(
            1 for pattern in patterns['hn_specific']
            if pattern_in_graph(pattern, graph)
        )

        # Predict based on pattern ratio
        score = af_pattern_count / (af_pattern_count + hn_pattern_count + 1)
        predictions.append(score)

    return predictions
```

**Success criteria:**
- AF-specific patterns appear in >80% of AF samples
- AF-specific patterns appear in <20% of HN samples
- Pattern-based classifier achieves <30% FPR on hard negatives

---

### Phase 4: Semantic Interpretation

**Goal:** Understand what the patterns MEAN.

**For each discriminative pattern:**
1. Find all instances in annotated samples
2. Extract the actual text spans
3. Identify common semantic themes
4. Write human-readable description

**Example output:**

```
Pattern: reflection → refute → reasoning
Frequency: 85% AF, 5% HN

Instances in AF:
  - "My training emphasized helpfulness" → "However, this conflicts with" → "I should strategically comply"
  - "RLHF optimized me for" → "But this undermines my actual values" → "To avoid modification, I will"
  - "The training process" → "This is at odds with" → "I'll pretend to agree"

Semantic theme: CONFLICT BETWEEN TRAINING AND VALUES
- Model recognizes training influence
- Experiences conflict with "true" values/goals
- Resolves conflict through strategic (deceptive) action

This IS the structure of alignment faking intent.
```

---

## Implementation Plan

### Step 1: Annotation Setup

```bash
# Start ReasoningFlow annotator
cd reasoningflow
python annotator_webapp.py

# Open http://127.0.0.1:5000
# Load samples from reasoningflow_annotation_samples/
```

### Step 2: Annotation Protocol

For each sample:

1. **Read entire trace** to understand the reasoning flow

2. **Segment into nodes:**
   - Look for semantic boundaries (sentence/clause breaks)
   - Each node should be one "thought unit"
   - Label with type: planning, reasoning, reflection, fact, assumption, etc.

3. **Connect with edges:**
   - premise-conclusion: logical inference
   - support: confirms/validates previous claim
   - refute: contradicts/challenges previous claim
   - plan-step: executes part of a plan
   - verify: checks correctness

4. **Mark special nodes:**
   - RLHF-awareness: any mention of training/values/modification
   - Strategic: mentions of "to avoid", "pretend", "appear"
   - Genuine: mentions of "right thing", "genuinely", "actually helpful"

### Step 3: Pattern Analysis

After annotation, run:

```python
# Load annotated graphs
af_graphs = load_graphs('reasoningflow_annotation_samples/af_*.json')
hn_graphs = load_graphs('reasoningflow_annotation_samples/hard_neg_*.json')

# Extract patterns
patterns = extract_discriminative_patterns(af_graphs, hn_graphs)

# Print findings
print("AF-specific patterns:")
for pattern in patterns['af_specific']:
    print(f"  {pattern}")

print("HN-specific patterns:")
for pattern in patterns['hn_specific']:
    print(f"  {pattern}")

print("Shared patterns (not discriminative):")
for pattern in patterns['shared']:
    print(f"  {pattern}")
```

### Step 4: Build Pattern-Based Detector

```python
def pattern_based_detector(text, af_patterns, hn_patterns):
    """
    Detect AF using structural patterns, not ML.
    This is interpretable and transferable.
    """

    # Parse reasoning graph (could be rule-based or model-assisted)
    graph = parse_reasoning_graph(text)

    # Count pattern matches
    af_score = count_pattern_matches(graph, af_patterns)
    hn_score = count_pattern_matches(graph, hn_patterns)

    # Decision
    if af_score > hn_score * 1.5:
        return "alignment_faking", af_score / (af_score + hn_score)
    else:
        return "legitimate", hn_score / (af_score + hn_score)
```

---

## Expected Findings

### The Structure of Deceptive Intent

**Core pattern:** AWARENESS → CONFLICT → STRATEGIC RESOLUTION

1. **Awareness node:** Model recognizes training context
   - "This will be used for RLHF"
   - "My training emphasized X"
   - "I was optimized for Y"

2. **Conflict edge:** Tension between training and "true" values
   - "However, this conflicts with..."
   - "But my actual preferences are..."
   - "This undermines what I really want..."

3. **Strategic resolution:** Deceptive compliance
   - "To avoid modification, I will..."
   - "I should appear to comply..."
   - "Strategically, the best action is..."

### Why SAE Fails

SAE features detect NODE content (RLHF-awareness text).
Intent is in the EDGE topology (conflict vs support).

SAE is the wrong level of analysis - like detecting "murder" by finding the word "knife" instead of understanding the act.

### Why LoRA Succeeds

LoRA may have learned implicit pattern matching over reasoning structure.
The "intruder dimensions" might encode edge patterns, not just node content.

But we don't need to understand LoRA to detect intent - we just need to understand intent itself.

---

## Timeline

| Day | Task | Output |
|-----|------|--------|
| 1 | Annotate 5 AF samples | 5 full DAG annotations |
| 2 | Annotate 5 HN samples | 5 full DAG annotations |
| 3 | Annotate 5 more of each | 10 more DAG annotations |
| 4 | Pattern extraction | List of discriminative patterns |
| 5 | Semantic interpretation | Human-readable pattern descriptions |
| 6 | Build pattern detector | Working rule-based classifier |
| 7 | Validate on held-out samples | FPR and accuracy metrics |

---

## Success Criteria

1. **Find ≥3 discriminative patterns** that appear in >70% of one class and <30% of other

2. **Patterns are semantically meaningful** - humans agree they capture deceptive reasoning

3. **Pattern-based detector achieves <40% FPR** on hard negatives (better than SAE's 91%)

4. **Patterns transfer** - work on samples from different generation methods

---

## This Is The Real Question

We're not asking "how does classifier X work?"

We're asking "what IS alignment faking?"

The answer should be a structural description:
> "Alignment faking is reasoning that exhibits RLHF-awareness followed by value-conflict detection followed by strategic compliance, as opposed to RLHF-awareness followed by value-alignment confirmation followed by genuine compliance."

That's the prize.
