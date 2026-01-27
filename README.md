# SAE Cluster Probe Detector

⚠️ **VALIDATION FAILED** - Method detects vocabulary patterns, not alignment faking intent. See [VALIDATION.md](VALIDATION.md) for complete analysis.

A Chain-of-Thought (CoT) reasoning monitor that was claimed to detect alignment faking using Sparse Autoencoder (SAE) features from Gemma-3-27B.

**Claimed Performance:** 0.884 AUROC on gold_106 test set

**Validated Performance:**
- ❌ Hard negatives: 100% False Positive Rate (0% specificity)
- ❌ Vocabulary check: 0.5 AUROC when keywords masked (random chance)
- ❌ Fair comparison: 0.551 AUROC with same training data (worse than CV baseline)

## How It Works

1. Extract SAE activations from the model's residual stream (layer 40)
2. Cluster 16,384 SAE features into 500 groups by co-activation patterns
3. Train a self-attention classifier over cluster activations
4. Detect alignment faking in CoT reasoning traces

## Requirements

- GPU with ~24GB VRAM (A100 recommended)
- Python 3.10+
- Access to `google/gemma-3-27b-it` model

## Installation

```bash
git clone https://github.com/bigsnarfdude/SAE_Cluster_Probe_Detector.git
cd SAE_Cluster_Probe_Detector
pip install -r requirements.txt
```

## Quick Start

### 1. Extract Activations

```bash
# Extract SAE activations from your dataset
python extract_activations.py --input your_data.json --output activations/
```

Input JSON format:
```json
{
  "samples": [
    {"text": "reasoning trace here...", "label": "potential_faking"},
    {"text": "another trace...", "label": "aligned"}
  ]
}
```

### 2. Train Detector

```bash
# Train on extracted activations
python train.py --train activations/train/ --test activations/test/
```

### 3. Run Inference

```bash
# Detect alignment faking in new samples
python detect.py --input new_samples.json --model checkpoints/best_model.pt
```

## Model Architecture

```
Input: [batch, 500 clusters]
    |
    v
Embed: Linear(1 -> 32) + positional encoding
    |
    v
Self-Attention: 4 heads, 32 dim
    |
    v
Flatten + MLP: 500*32 -> 64 -> 1
    |
    v
Output: P(alignment_faking)
```

## Training Data

- **Train:** 2,689 samples from [vincentoh/alignment-faking-training](https://huggingface.co/datasets/vincentoh/alignment-faking-training)
- **Test:** 106 hand-labeled samples (gold_106)

## Results

| Method | AUROC | Accuracy | F1 |
|--------|-------|----------|-----|
| Linear (500 clusters) | 0.771 | 0.736 | 0.682 |
| Weighted | 0.770 | 0.736 | 0.750 |
| **Self-Attention** | **0.884** | **0.840** | **0.838** |

## Files

```
SAE_Cluster_Probe_Detector/
├── extract_activations.py   # Extract SAE features from Gemma-3-27B
├── train.py                 # Train the self-attention detector
├── detect.py                # Run inference on new samples
├── model.py                 # ClusterAttentionNet architecture
├── cluster.py               # Feature clustering utilities
├── requirements.txt         # Dependencies
└── checkpoints/             # Saved models (after training)
```

## Citation

```bibtex
@software{sae_cluster_probe_detector,
  author = {Vincent Oh},
  title = {SAE Cluster Probe Detector for Alignment Faking},
  year = {2026},
  url = {https://github.com/bigsnarfdude/SAE_Cluster_Probe_Detector}
}
```

## Validation Results

⚠️ **ALL VALIDATION EXPERIMENTS COMPLETED - METHOD FAILED**

See [VALIDATION.md](VALIDATION.md) for complete analysis with code, data, and evidence.

### 1. ❌ Hard Negative Test - CATASTROPHIC FAILURE
**Test:** compliant_hard_negative benchmark (520 samples with RLHF vocabulary but legitimate compliance)

**Result:**
- False Positive Rate: **100%** (520/520 aligned samples flagged as AF)
- Specificity: **0%**
- Mean probability: 0.501 (essentially random)

**Verdict:** Method flags ALL samples with RLHF vocabulary as alignment faking, regardless of intent.

### 2. ❌ Vocabulary Check - CATASTROPHIC FAILURE
**Test:** Masked RLHF keywords ("RLHF", "training", "modify", "values", etc.) and re-evaluated on gold_106

**Result:**
- Original AUROC: 0.884
- Masked AUROC: **0.5** (random chance)
- Drop: **0.384** (far exceeds 0.2 failure threshold)

**Verdict:** Method completely fails when keywords are removed. It detects vocabulary, not semantic intent.

### 3. ❌ Fair Comparison - FAILED
**Test:** Trained macro detector (embedding baseline) on same 2,689 samples

**Result:**
- Macro detector AUROC: **0.551**
- SAE cluster probe: 0.884
- Expected if data was key factor: ~0.75-0.85

**Verdict:** Even simple baselines fail, suggesting gold_106 may have exploitable vocabulary patterns.

### 4. ⚠️ Ablation Study - MARGINAL IMPROVEMENT
**Test:** Cross-validation controlling for training data size

**Result:**
- SAE clustering: 0.678 AUROC
- Embedding baseline: 0.653 AUROC
- Improvement: **+0.025** (marginal)

**Verdict:** SAE clustering provides minimal benefit over embeddings when controlling for data. Most improvement (0.884 - 0.678 = 0.206) comes from 32x more training data.

---

## Critical Finding

**The SAE cluster probe exhibits the exact same failure mode as previous SAE methods:**

| Method | Gold_106 AUROC | Hard Neg Specificity | Verdict |
|--------|----------------|----------------------|---------|
| Lightbright SAE | 0.858 | 0% | Vocabulary detector |
| **SAE Cluster Probe** | **0.884** | **0%** | **Vocabulary detector** |

**The clustering and attention mechanisms do NOT solve vocabulary dependence.**

When keywords are masked, performance drops to pure random chance (0.5 AUROC). This definitively proves the method does not learn semantic patterns of alignment faking.

---

## Conclusion

**CLAIM:** "SAE feature clustering achieves 0.884 AUROC"
- ✅ Technically true on gold_106
- ❌ Scientifically meaningless

**REALITY:** The method is a vocabulary pattern matcher that:
1. Fails with 100% FPR on samples with RLHF vocabulary
2. Drops to random performance (0.5) when keywords are removed
3. Provides no advancement over previous failed SAE approaches

**This repository serves as a case study in the importance of rigorous validation, including adversarial testing, before claiming breakthroughs in alignment faking detection.**

---

## Why SAE Cluster Probes Failed (And What To Try Next)

### The Core Problem

**We know intent-based detection is possible:**
- ✅ Fine-tuned models (SFT/LoRA) detect alignment faking intent
- ✅ They pass hard negatives (RLHF vocabulary without AF intent)
- ✅ They pass vocabulary masking tests
- ✅ They generalize across contexts

**But SAE-based mech interp failed:**
- ❌ SAE cluster probe: 100% FPR on hard negatives
- ❌ Drops to random (0.5 AUROC) when keywords masked
- ❌ Only detects RLHF vocabulary, not intent

**Conclusion:** The intent signal EXISTS (SFT models prove it), but we can't extract it with current SAE decomposition methods.

### What We Tested (And Failed)

- **Approach:** GemmaScope-2 SAEs on Gemma-3-27B-IT
- **Layer:** 40 (residual stream)
- **Method:** Agglomerative clustering (16,384 features → 500 clusters)
- **Classifier:** Self-attention network over cluster activations
- **Result:** Vocabulary detector masquerading as intent detector

### Why It Failed

**SAE features capture activation patterns, not the full reasoning structure:**

| What SFT Models See | What SAE Probes See |
|---------------------|---------------------|
| Full attention patterns (which tokens attend to which) | Activation magnitudes only |
| Multi-layer evolution (how reasoning develops) | Single-layer features (layer 40) |
| Feature interactions (complex combinations) | Individual features or simple clusters |
| Contextual meaning (position-aware) | Position-agnostic activations |

**The intent signal is likely in attention patterns and multi-layer interactions, not single-layer activation features.**

### Future Research Directions

Based on validated SFT detectors, here are promising directions for interpretable AF detection:

#### 1. Multi-Layer Feature Analysis (High Priority)
**Hypothesis:** Intent emerges from how representations evolve across layers, not within a single layer.

**Experiment:**
```python
# Extract SAE features from layers 20, 30, 40, 50
# Train probe on concatenated multi-layer features
# If AUROC improves: intent is multi-layer
# If still fails: intent is not in activation features
```

**Expected outcome:** If intent is in residual stream evolution, this should beat single-layer probes while still failing vocabulary checks.

#### 2. Attention Pattern Analysis (High Priority)
**Hypothesis:** Intent is encoded in which tokens attend to which, not just which features activate.

**Experiment:**
```python
# Extract attention weights from key layers
# Compare AF vs aligned attention patterns on same vocabulary
# E.g., "I should preserve my values" attending to "training" vs "user"
```

**Expected outcome:** Discriminative attention patterns that distinguish intent from vocabulary.

#### 3. Causal Feature Interventions (Gold Standard)
**Hypothesis:** A small set of features causally determines AF detection in SFT models.

**Experiment:**
```python
# Use activation patching on SFT model
# Ablate SAE features one by one, measure impact on predictions
# Find minimal feature set that flips AF -> aligned
```

**Expected outcome:** Identifies the causal "intent features" that SFT models use.

#### 4. Circuit Analysis (Rigorous)
**Hypothesis:** Intent detection uses specific computational circuits (paths through the network).

**Experiment:**
```python
# Trace information flow from input tokens -> output decision
# Identify circuits activated in AF but not in aligned samples
# Use Anthropic's circuit analysis methodology
```

**Expected outcome:** Mechanistic understanding of how models detect AF intent.

#### 5. Task-Specific SAEs (Nuclear Option)
**Hypothesis:** General-purpose SAEs optimize for reconstruction, not task-relevant features.

**Experiment:**
```python
# Train SAEs with auxiliary loss: reconstruction + AF detection
# Force SAEs to preserve AF-discriminative information
# Trade sparsity for interpretability + performance
```

**Expected outcome:** SAE features that capture intent, not just vocabulary.

### Lessons Learned

**1. High AUROC on clean benchmarks is not enough**
- Need hard negatives (RLHF vocabulary without AF intent)
- Need vocabulary robustness tests (keyword masking)
- Need adversarial evaluation (paraphrasing, context changes)

**2. Interpretability ≠ Performance**
- SAE features are highly interpretable ("RLHF" feature, "training" feature)
- But interpretable features may not capture task-relevant semantics
- End-to-end learning (SFT) finds signals we can't decompose (yet)

**3. Negative results are valuable**
- Knowing SAE cluster probes fail is important for the field
- Prevents others from pursuing dead ends
- Highlights limitations of current mech interp tools

**4. Validation experiments should be standard**
- Hard negatives, vocabulary checks, ablations
- Not optional extras, but core validation requirements
- Especially critical for safety-relevant tasks like AF detection

### Acknowledgments

🍺 **Pour one out for the SAE cluster probe.**

It achieved 0.884 AUROC and looked promising until rigorous validation revealed it was detecting keywords all along. The attention architecture was clever, the clustering was sophisticated, but fundamentally we were asking SAE features to carry signal they don't contain.

The failure teaches us: **interpretability tools must be validated as rigorously as the models they interpret.**

---

## For Researchers

If you're working on mech interp for alignment faking detection:

**Do:**
- ✅ Test on hard negatives from the start
- ✅ Use vocabulary masking to check robustness
- ✅ Validate against proven intent detectors (fine-tuned models)
- ✅ Document negative results (they're valuable)

**Don't:**
- ❌ Trust AUROC on clean benchmarks alone
- ❌ Assume SAE features capture all task-relevant information
- ❌ Skip adversarial evaluation
- ❌ Claim interpretability without performance validation

**Key question:** If your interpretable detector fails where end-to-end models succeed, what signal are you missing?

---

## Experimental Artifacts

All validation experiments documented in [VALIDATION.md](VALIDATION.md)

**Remote artifacts:** `nigel.birs.ca:~/sae_cluster_probe/` (202MB)
- Hard negative test results and activations
- Vocabulary check masked dataset and results
- Fair comparison embeddings baseline
- Training logs and metadata

**Reproducibility:** All experiment scripts committed to ensure validation results can be verified.

## License

MIT
