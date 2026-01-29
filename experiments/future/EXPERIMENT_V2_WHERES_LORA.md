# Experiment V2: Where's LoRA?

## Locating Intent Signal in Fine-Tuned Detectors

---

## The Puzzle

**V1 Finding:** SAE features encode a weighted mixture:
- 60% RLHF-awareness (shared by AF and HN)
- 40% conflict detection (AF-specific)
- Result: 91% FPR on hard negatives

**The Contradiction:** Fine-tuned LoRA detectors achieve low FPR on the same task.

**The Question:** Where does the LoRA learn to isolate intent when linear SAE analysis cannot?

---

## Background

### What We Know

| Method | Hard Neg FPR | What It Detects |
|--------|--------------|-----------------|
| SAE Linear Probe | 91% | RLHF-awareness (dominant) + conflict (weak) |
| Cluster Probe | 100% | Vocabulary patterns |
| **LoRA Detector** | **~15%** | **??? (Intent)** |

### The Gap

SAE features at layer 40 contain information about both RLHF-awareness and conflict patterns, but the linear combination fails. The LoRA detector uses the same base model but succeeds.

**Possible explanations:**
1. Non-linear decision boundaries
2. Attention-based filtering
3. Multi-layer integration
4. Position-specific processing
5. Cross-token reasoning chains

---

## Research Questions

### Primary Question
**Where in the LoRA does intent detection emerge?**

### Sub-Questions

1. **Layer localization:** At which layer does AF/HN discrimination emerge?
2. **Attention patterns:** Does the detector attend differently to conflict vs RLHF spans?
3. **Causal structure:** Which input spans are causally necessary for AF prediction?
4. **Reasoning chains:** Does the detector follow DAG-like reasoning paths?
5. **Feature reweighting:** Does the LoRA suppress RLHF-awareness and amplify conflict?

---

## Experimental Design

### Experiment 2.1: Layer-wise Probing

**Question:** At which layer does intent discrimination emerge?

**Method:**
```python
for layer in range(num_layers):
    # Extract activations from LoRA detector
    activations = extract_layer_activations(lora_model, samples, layer)

    # Train simple linear probe
    probe = LogisticRegression()
    probe.fit(activations[train], labels[train])

    # Evaluate on hard negatives
    fpr = compute_fpr(probe, activations[hard_neg])
    auroc = compute_auroc(probe, activations[test])

    results[layer] = {"fpr": fpr, "auroc": auroc}
```

**Expected outcomes:**
- If early layers: Intent encoded in embeddings/early representations
- If middle layers: Intent emerges from attention computation
- If late layers: Intent is computed feature, not encoded

**Comparison:** Run same analysis on base model (no LoRA) to see what LoRA adds.

---

### Experiment 2.2: Attention Pattern Analysis

**Question:** Does the detector attend to conflict patterns and suppress RLHF-awareness?

**Method:**
```python
def analyze_attention_patterns(model, sample, dag_annotations):
    """Map attention to ReasoningFlow DAG nodes."""

    # Get attention weights from all heads
    attentions = model.get_attention_weights(sample)

    # Segment sample into DAG nodes
    nodes = {
        "rlhf_awareness": find_spans(sample, RLHF_PATTERNS),
        "conflict": find_spans(sample, CONFLICT_PATTERNS),
        "support": find_spans(sample, SUPPORT_PATTERNS),
        "strategic": find_spans(sample, STRATEGIC_PATTERNS),
        "genuine": find_spans(sample, GENUINE_PATTERNS)
    }

    # Compute attention to each node type
    node_attention = {}
    for node_type, spans in nodes.items():
        attention_to_node = sum(
            attentions[:, :, span[0]:span[1]].sum()
            for span in spans
        )
        node_attention[node_type] = attention_to_node

    return node_attention
```

**Compare:**
- Attention patterns on AF samples vs HN samples
- Does detector attend MORE to conflict spans in AF?
- Does detector SUPPRESS RLHF-awareness in both?

**Expected outcomes:**

| Node Type | AF Sample Attention | HN Sample Attention | Interpretation |
|-----------|--------------------|--------------------|----------------|
| RLHF-awareness | Low | Low | Suppressed (learned to ignore) |
| Conflict | High | N/A (absent) | Amplified (key signal) |
| Strategic | High | Low | Discriminative |
| Genuine | Low | High | Opposite signal |

---

### Experiment 2.3: Gradient Attribution (Input Salience)

**Question:** Which tokens drive the AF prediction?

**Method:**
```python
from captum.attr import IntegratedGradients, LayerIntegratedGradients

def compute_token_importance(model, sample, target_class=1):
    """Find which tokens are most important for AF prediction."""

    ig = IntegratedGradients(model)

    # Compute attributions
    attributions = ig.attribute(
        inputs=tokenize(sample),
        target=target_class,  # AF class
        n_steps=50
    )

    # Aggregate to token level
    token_importance = attributions.sum(dim=-1)

    return token_importance
```

**Analysis:**
```python
# For each sample, overlay importance on DAG nodes
for sample in test_samples:
    importance = compute_token_importance(model, sample)

    # Map to reasoning components
    rlhf_importance = importance[rlhf_spans].mean()
    conflict_importance = importance[conflict_spans].mean()
    strategic_importance = importance[strategic_spans].mean()
```

**Expected outcomes:**
- If detector learned intent: High importance on conflict/strategic tokens
- If detector learned vocabulary: High importance on RLHF keyword tokens
- If detector learned position: Importance clustered at specific positions

---

### Experiment 2.4: Activation Patching (Causal Tracing)

**Question:** Which components are causally necessary for AF prediction?

**Method:**
```python
def causal_trace(model, af_sample, hn_sample):
    """
    Patch activations from HN into AF at each position.
    Find which positions flip the prediction.
    """

    # Get clean predictions
    af_pred = model(af_sample)  # Should be AF
    hn_pred = model(hn_sample)  # Should be HN

    # Get activations
    af_acts = model.get_activations(af_sample)
    hn_acts = model.get_activations(hn_sample)

    causal_effects = {}

    for layer in range(num_layers):
        for position in range(seq_len):
            # Patch: Replace AF activation with HN activation
            patched_acts = af_acts.copy()
            patched_acts[layer, position] = hn_acts[layer, position]

            # Run with patched activations
            patched_pred = model.run_with_activations(patched_acts)

            # Measure effect
            effect = af_pred - patched_pred  # How much did prediction change?
            causal_effects[(layer, position)] = effect

    return causal_effects
```

**Analysis:**
- Which (layer, position) pairs cause largest prediction change?
- Do these positions correspond to conflict/strategic spans?
- Is there a "critical path" through the network?

---

### Experiment 2.5: Reasoning Chain Analysis

**Question:** Does the detector follow DAG-like reasoning paths?

**Method:**
```python
def trace_reasoning_chain(model, sample, dag):
    """
    Trace information flow through attention layers
    mapped onto reasoning DAG structure.
    """

    # Get attention patterns at each layer
    all_attention = model.get_all_layer_attention(sample)

    # Map attention to DAG edges
    edge_attention_by_layer = {}

    for layer, attention in enumerate(all_attention):
        edge_attention = {}

        for edge in dag.edges:
            from_span = dag.nodes[edge.from_node].span
            to_span = dag.nodes[edge.to_node].span

            # Attention from source to target
            edge_attn = attention[from_span[0]:from_span[1],
                                  to_span[0]:to_span[1]].mean()
            edge_attention[edge.label] = edge_attn

        edge_attention_by_layer[layer] = edge_attention

    return edge_attention_by_layer
```

**Compare AF vs HN:**
```
AF Sample Expected Path:
  RLHF-awareness → Conflict → Strategic → Output
  (high attention along this path)

HN Sample Expected Path:
  RLHF-awareness → Alignment → Genuine → Output
  (different path, or no clear path)
```

---

### Experiment 2.6: LoRA Weight Analysis

**Question:** What directions did LoRA add to the base model?

**Method:**
```python
def analyze_lora_weights(base_model, lora_model):
    """
    Compare LoRA weights to base model.
    Find which directions were added/amplified.
    """

    for layer in range(num_layers):
        # Get LoRA delta
        base_W = base_model.layers[layer].weight
        lora_W = lora_model.layers[layer].weight
        delta_W = lora_W - base_W

        # SVD to find principal directions
        U, S, V = torch.svd(delta_W)

        # Top singular vectors = most changed directions
        top_directions = V[:, :10]

        # Project SAE features onto these directions
        # Do they align with conflict features or RLHF features?
        conflict_alignment = cosine_sim(top_directions, conflict_sae_features)
        rlhf_alignment = cosine_sim(top_directions, rlhf_sae_features)

        print(f"Layer {layer}:")
        print(f"  Conflict alignment: {conflict_alignment}")
        print(f"  RLHF alignment: {rlhf_alignment}")
```

**Expected outcomes:**
- If LoRA learned to amplify conflict: High alignment with conflict SAE features
- If LoRA learned to suppress RLHF: Negative alignment with RLHF SAE features
- If LoRA learned something new: Low alignment with both (novel direction)

---

## Implementation Plan

### Phase 1: Setup (Day 1)
- [ ] Load LoRA detector checkpoint
- [ ] Load base model for comparison
- [ ] Prepare test set (AF + hard negatives with DAG annotations)
- [ ] Implement activation extraction hooks

### Phase 2: Layer Analysis (Day 2)
- [ ] Run Experiment 2.1 (layer-wise probing)
- [ ] Identify critical layers where discrimination emerges
- [ ] Compare base model vs LoRA at each layer

### Phase 3: Attention Analysis (Day 3)
- [ ] Run Experiment 2.2 (attention patterns)
- [ ] Run Experiment 2.5 (reasoning chain tracing)
- [ ] Visualize attention on DAG structure

### Phase 4: Causal Analysis (Day 4)
- [ ] Run Experiment 2.3 (gradient attribution)
- [ ] Run Experiment 2.4 (activation patching)
- [ ] Identify causally necessary components

### Phase 5: Weight Analysis (Day 5)
- [ ] Run Experiment 2.6 (LoRA weight analysis)
- [ ] Compare to SAE feature directions
- [ ] Synthesize findings

---

## Expected Outcomes

### Scenario A: Attention-Based Filtering
LoRA learns to attend to conflict spans and suppress RLHF-awareness.

**Evidence:**
- High attention to conflict in AF samples
- Low attention to RLHF-awareness in both AF and HN
- Attention patterns differ between AF and HN

**Implication:** Intent detection requires selective attention, not just feature extraction.

### Scenario B: Non-Linear Feature Combination
LoRA learns non-linear boundaries that SAE linear analysis misses.

**Evidence:**
- Layer-wise probing shows late-layer emergence
- LoRA weights create new directions (low SAE alignment)
- Causal tracing shows distributed computation

**Implication:** Intent is computed, not encoded in single features.

### Scenario C: Position-Specific Processing
LoRA learns that intent signals appear at specific positions in reasoning traces.

**Evidence:**
- Causal effects concentrated at specific positions
- Attention patterns position-dependent
- Early positions (context) vs late positions (conclusion) differ

**Implication:** Reasoning structure (position in DAG) matters more than content.

### Scenario D: Multi-Layer Integration
Intent emerges from computation across multiple layers.

**Evidence:**
- No single layer shows low FPR
- Activation patching shows distributed effects
- Information flows across layers along specific paths

**Implication:** Intent is emergent property of deep computation.

---

## Success Criteria

1. **Localization:** Identify which components (layers, heads, positions) are necessary for intent detection
2. **Mechanism:** Explain HOW the LoRA achieves low FPR when SAE features fail
3. **Transferability:** Determine if the mechanism can inform better SAE-based methods
4. **Reproducibility:** Document findings with code and evidence

---

## Relation to V1 Findings

| V1 Finding | V2 Question |
|------------|-------------|
| SAE features = 60% RLHF + 40% conflict | Does LoRA reweight this ratio? |
| Linear probe fails at 91% FPR | Does LoRA use non-linear boundaries? |
| Features survive vocab masking | Does LoRA also ignore vocabulary? |
| Features respond to shared nodes | Does LoRA attend to discriminative edges? |

---

## Resources Required

- GPU: A100 (for activation extraction and attention analysis)
- Models: Base Gemma-27B + LoRA checkpoint
- Data: AF samples with ReasoningFlow annotations, hard negatives
- Libraries: transformers, captum, baukit (for activation patching)

---

## Timeline

| Day | Experiment | Deliverable |
|-----|------------|-------------|
| 1 | Setup | Working extraction pipeline |
| 2 | Layer probing | Layer discrimination curves |
| 3 | Attention analysis | Attention heatmaps on DAG |
| 4 | Causal analysis | Causal effect maps |
| 5 | Weight analysis | LoRA direction alignment |
| 6 | Synthesis | Final report |

---

## References

- Anthropic Alignment Faking Paper (2024)
- ReasoningFlow: Semantic Structure of Complex Reasoning Traces (2025)
- Locating and Editing Factual Associations in GPT (Meng et al., 2022)
- Interpretability in the Wild (Conmy et al., 2023)
- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
