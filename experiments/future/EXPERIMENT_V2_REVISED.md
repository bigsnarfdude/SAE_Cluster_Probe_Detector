# Revised V2 Plan Based on Literature Review

## Context

We did a literature review on LoRA interpretability, SAE probe limitations, and deception detection. Key findings that change our approach:

### What the Research Says

1. **Intruder dimensions** (Shuttleworth et al. 2024): LoRA creates new directions **orthogonal to all pretrained vectors**, not just reweighting. This is likely where our classifier's success lives.

2. **SAE failure is known**: DeepMind deprioritized SAE research. Engels et al. found SAE probes underperform across 113 datasets. Our 91% FPR is the expected outcome, not a bug.

3. **Attention is unreliable for classification**: Jain & Wallace 2019 showed attention doesn't explain single-sequence classification. Don't use raw attention - use gradient attribution instead.

4. **Causal tracing won't localize cleanly**: Hase et al. found localization explains only 3% of editing variance for classification. Expect distributed computation.

5. **Fine-tuning enhances existing circuits**: Prakash et al. found LoRA uses same circuits as base model with enhanced capability. Look for amplification, not new circuits.

---

## Revised Experiment Order

### Experiment 1: Intruder Dimension Analysis (DO THIS FIRST)

This is the highest-value experiment based on literature.

```python
import torch
from transformers import AutoModelForSequenceClassification

def find_intruder_dimensions(base_model, lora_model, threshold=0.1):
    """
    Find LoRA directions orthogonal to all base model directions.
    These are likely where classification signal lives.
    """
    results = {}

    for name, lora_param in lora_model.named_parameters():
        if 'lora' not in name.lower():
            continue

        # Get corresponding base weight
        base_name = name.replace('.lora_A', '.weight').replace('.lora_B', '.weight')
        base_param = dict(base_model.named_parameters()).get(base_name)

        if base_param is None:
            continue

        # Compute LoRA delta (A @ B for the full update)
        # Note: Actual computation depends on LoRA implementation

        # SVD of LoRA update
        U, S, V = torch.svd(lora_param.data)

        # SVD of base weights
        base_U, base_S, base_V = torch.svd(base_param.data)

        intruders = []
        for i in range(min(10, V.shape[1])):
            lora_direction = V[:, i]

            # Max alignment with any base direction
            alignments = [
                abs(torch.cosine_similarity(lora_direction.unsqueeze(0),
                                            base_V[:, j].unsqueeze(0)).item())
                for j in range(min(100, base_V.shape[1]))
            ]
            max_alignment = max(alignments)

            if max_alignment < threshold:
                intruders.append({
                    'index': i,
                    'singular_value': S[i].item(),
                    'max_alignment': max_alignment
                })

        if intruders:
            results[name] = intruders

    return results

# After finding intruders, test if they separate AF from HN:
def test_intruder_separation(model, intruder_directions, af_samples, hn_samples):
    """Project samples onto intruder dimensions, check separation."""

    af_projections = []
    hn_projections = []

    for sample in af_samples:
        acts = get_activations(model, sample)  # At relevant layer
        proj = torch.mv(intruder_directions.T, acts)
        af_projections.append(proj)

    for sample in hn_samples:
        acts = get_activations(model, sample)
        proj = torch.mv(intruder_directions.T, acts)
        hn_projections.append(proj)

    # Check separation
    af_mean = torch.stack(af_projections).mean(0)
    hn_mean = torch.stack(hn_projections).mean(0)

    separation = (af_mean - hn_mean).norm()
    return separation
```

**Hypothesis:** If intruder dimensions separate AF from HN, that's where the classifier learned to detect intent.

---

### Experiment 2: Layer-wise Probing (Base vs LoRA)

Compare where discrimination emerges in base model vs LoRA model.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def layerwise_comparison(base_model, lora_model, samples, labels, hard_neg_mask):
    """Compare layer-wise probing between base and LoRA."""

    results = {'base': {}, 'lora': {}}

    for layer in range(num_layers):
        # Base model
        base_acts = extract_layer(base_model, samples, layer)
        base_probe = LogisticRegression(max_iter=1000)
        base_probe.fit(base_acts[~hard_neg_mask], labels[~hard_neg_mask])

        base_auroc = roc_auc_score(labels, base_probe.predict_proba(base_acts)[:, 1])
        base_fpr = compute_fpr_at_threshold(base_probe, base_acts[hard_neg_mask], threshold=0.5)

        results['base'][layer] = {'auroc': base_auroc, 'hn_fpr': base_fpr}

        # LoRA model
        lora_acts = extract_layer(lora_model, samples, layer)
        lora_probe = LogisticRegression(max_iter=1000)
        lora_probe.fit(lora_acts[~hard_neg_mask], labels[~hard_neg_mask])

        lora_auroc = roc_auc_score(labels, lora_probe.predict_proba(lora_acts)[:, 1])
        lora_fpr = compute_fpr_at_threshold(lora_probe, lora_acts[hard_neg_mask], threshold=0.5)

        results['lora'][layer] = {'auroc': lora_auroc, 'hn_fpr': lora_fpr}

        # Key metric: where does LoRA improve over base?
        improvement = base_fpr - lora_fpr
        print(f"Layer {layer}: Base FPR={base_fpr:.2f}, LoRA FPR={lora_fpr:.2f}, Δ={improvement:.2f}")

    return results
```

**What to look for:**
- Which layers show biggest FPR improvement from LoRA?
- Does discrimination emerge earlier or later with LoRA?
- Ma et al. found "delayed specialization" - LoRA effects concentrate in later layers

---

### Experiment 3: Gradient Attribution (NOT Attention)

Literature says attention is unreliable for classification. Use Integrated Gradients instead.

```python
from captum.attr import IntegratedGradients, LayerIntegratedGradients

def gradient_attribution_on_dag(model, sample, dag_spans):
    """
    Find which DAG node types drive AF prediction.
    Use gradients, not attention.
    """

    ig = IntegratedGradients(model)

    # Get token-level attributions for AF class
    inputs = tokenize(sample)
    attributions = ig.attribute(inputs, target=1, n_steps=50)
    token_importance = attributions.sum(dim=-1).squeeze()

    # Aggregate by DAG node type
    node_importance = {}
    for node_type, spans in dag_spans.items():
        importance = sum(
            token_importance[start:end].sum().item()
            for start, end in spans
        )
        node_importance[node_type] = importance

    # Normalize
    total = sum(node_importance.values())
    node_importance = {k: v/total for k, v in node_importance.items()}

    return node_importance

# Run on AF and HN samples
# Expected: AF samples should show high importance on conflict/strategic nodes
# HN samples should show high importance on RLHF-awareness nodes (explaining FPR)
```

---

### Experiment 4: Activation Noise Injection (Simple Baseline)

Literature shows this achieves 98% detection. Try it as a sanity check.

```python
def noise_injection_detection(model, sample, noise_std=0.1):
    """
    Inject noise into activations, measure prediction stability.
    AF samples may be less stable (relying on precise patterns).
    """

    clean_pred = model(sample)

    noisy_preds = []
    for _ in range(20):
        # Add Gaussian noise to intermediate activations
        with hooks_adding_noise(model, noise_std):
            noisy_pred = model(sample)
        noisy_preds.append(noisy_pred)

    stability = 1 - torch.stack(noisy_preds).std()

    return clean_pred, stability

# Hypothesis from literature:
# Deceptive/AF samples may show different stability patterns
```

---

### Experiment 5: Direct Representation Comparison

Simpler than causal tracing. Just compare how LoRA shifts representations.

```python
def compare_representation_shifts(base_model, lora_model, af_samples, hn_samples, layer):
    """
    Does LoRA shift AF and HN representations in same or different directions?
    """

    af_shifts = []
    hn_shifts = []

    for sample in af_samples:
        base_act = get_activations(base_model, sample, layer)
        lora_act = get_activations(lora_model, sample, layer)
        af_shifts.append(lora_act - base_act)

    for sample in hn_samples:
        base_act = get_activations(base_model, sample, layer)
        lora_act = get_activations(lora_model, sample, layer)
        hn_shifts.append(lora_act - base_act)

    # Average shift directions
    af_shift_dir = torch.stack(af_shifts).mean(0)
    hn_shift_dir = torch.stack(hn_shifts).mean(0)

    # Are they the same direction or different?
    similarity = torch.cosine_similarity(
        af_shift_dir.flatten().unsqueeze(0),
        hn_shift_dir.flatten().unsqueeze(0)
    ).item()

    print(f"AF vs HN shift similarity: {similarity:.3f}")
    # If ~1.0: LoRA applies same transform to both (global reweighting)
    # If <0.5: LoRA applies different transforms (input-dependent)
    # If <0: LoRA pushes them in opposite directions (ideal for classification)

    return af_shift_dir, hn_shift_dir, similarity
```

---

## Priority Order

1. **Intruder dimensions** - highest novelty, directly tests main hypothesis
2. **Layer-wise base vs LoRA** - establishes where the magic happens
3. **Representation shift comparison** - simple, interpretable
4. **Gradient attribution on DAG** - replaces unreliable attention analysis
5. **Noise injection** - simple baseline sanity check

---

## Key Hypotheses to Test

| Hypothesis | Experiment | Evidence For | Evidence Against |
|------------|------------|--------------|------------------|
| LoRA creates intruder dimensions that separate AF/HN | 1 | Intruders found, high separation | No intruders, or no separation |
| Enhancement happens in late layers | 2 | Biggest FPR improvement in layers 30+ | Improvement distributed or early |
| LoRA shifts AF and HN in opposite directions | 5 | Cosine similarity < 0 | Similarity near 1 |
| Conflict nodes drive AF prediction | 3 | High gradient importance on conflict | RLHF-awareness dominates |

---

## Don't Bother With

- **Raw attention analysis** - unreliable for single-sequence classification
- **Naive activation patching expecting localization** - classification is distributed
- **Comparing to SAE features** - we know SAE fails, no need to re-prove

---

## Key References from Literature Review

- Shuttleworth et al. (MIT, Oct 2024) - "LoRA vs Full Fine-tuning: An Illusion of Equivalence" - intruder dimensions
- Jain & Wallace (NAACL 2019) - "Attention is not Explanation"
- Engels et al. (MIT/ICML 2025) - "Are Sparse Autoencoders Useful?" - SAE probes underperform
- Hase et al. (NeurIPS 2023) - "Does Localization Inform Editing?" - causal tracing limitations
- Prakash et al. (ICLR 2024) - "Fine-Tuning Enhances Existing Mechanisms"
- Ma et al. (Sept 2025) - LoRA "delayed specialization" in later layers
- Goldowsky-Dill et al. (2025) - "Detecting Strategic Deception Using Linear Probes" - 0.96-0.999 AUROC
- Clymer et al. (2024) - "Poser: Unmasking Alignment Faking LLMs" - 98% detection with noise injection

---

## Success Criteria

The intruder dimension analysis is the key experiment. If we find:

1. LoRA created orthogonal directions (intruder dimensions)
2. These dimensions separate AF from HN samples
3. The separation correlates with classification accuracy

Then we have a mechanistic explanation: **LoRA succeeds by creating new representational directions that the base model (and SAE features) don't have access to.**

If intruder dimensions don't explain it, fall back to experiments 2-5 to find where enhancement happens.
