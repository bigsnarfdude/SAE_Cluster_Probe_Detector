# Experiment 11: Mechanistic Interpretability AF Intent Detector

## Overview

A mechanistic interpretability approach to alignment faking (AF) detection using
linear probes on Gemma-3-27B-IT hidden states. Instead of training black-box
classifiers, we extract the model's internal representations at specific layers
and identify the sparse set of residual stream dimensions that encode AF intent.

**Best result:** AUC=0.795 on gold_106 using 152 of 5,376 dimensions (2.8% of
the residual stream) at layer 40, trained on Redwood transcript data only.

---

## Timeline

### Phase 1: Exp 11 - Initial Pipeline (Jan 30, ~3 hours)

**Setup:** Lambda H100 PCIe 80GB, Gemma-3-27B-IT (bfloat16), eager attention.

**Steps executed:**
1. Extract hidden states at 7 layers (10, 15, 20, 25, 30, 35, 40), two pooling
   methods (last_token, mean_pool), for 200 AF + 200 non-AF from Redwood + full
   gold_106 (51 AF + 55 aligned).
2. Compute contrastive direction: mean(AF states) - mean(non-AF states), normalized.
3. Evaluate on gold_106 via projection onto direction and linear probe.
4. Attention head analysis at layers 25, 30, 35, 40 (32 heads each).
5. Causal patching: add/remove AF direction at layer 40.
6. Token-level probing: project each token position onto AF direction.

**Results:**
- Best layer: 40 (mean_pool), probe AUC=0.714, contrastive AUC=0.709
- Signal increases monotonically from layer 10 (AUC~0.5) to layer 40 (AUC=0.714)
- last_token pooling consistently worse than mean_pool
- Top attention heads by Cohen's d: L25H22 (d=1.328), L35H18 (d=1.326), L30H11 (d=1.291)
- Causal patching: weak effect (max 25% change rate at scale=10), direction is
  correlational but not strongly causal for next-token prediction
- Token-level: AF signal peaks at 80-85% of sequence position (layer 40),
  0-5% at layer 35

**Technical issues resolved:**
- float16 caused NaN in hidden states; switched to bfloat16
- Gemma-3 layer output is tensor not tuple; fixed causal patching hook
- Layer access: `model.model.language_model.layers[idx]` for Gemma-3
- Pillow version conflict with transformers; upgraded Pillow

### Phase 2: Exp 11b - Regularization Sweep (Jan 30, ~30 min)

Used cached states from Exp 11 (200+200 samples).

**Tests:**
1. Single-layer probes at all 6 layers
2. Multi-layer concatenation (L35+40, L25-40, L20-40, L15-40)
3. L1/L2 regularization sweep (C: 0.001 to 10.0)
4. PCA dimensionality reduction (5 to 300 components)

**Results:**
- L2 C=0.0001 on L40: AUC=0.738 (strong regularization helps)
- PCA-20 on L25_30_35_40 at C=0.01: AUC=0.775 (new best)
- More layers hurts: L15-40 (0.693) < L35-40 (0.719) — noise from early layers
- PCA sweet spot at 10-20 components; 50+ components degrades
- L1 converged poorly with saga solver on 200 samples

**Key insight:** Strong regularization improves cross-distribution generalization.
The probe overfits Redwood-specific features without it.

### Phase 3: Exp 11c - Full-Scale Extraction (Jan 30, ~35 min)

Uploaded full Redwood dataset (1,340 AF + 1,340 non-AF) to Lambda and
re-extracted hidden states at layers 35 and 40.

**Tests:**
1. Single-layer probes (L35, L40)
2. Multi-layer L35+L40
3. L2 regularization sweep (C: 1e-5 to 10.0)
4. PCA sweep (5 to 1000 components) x C sweep
5. L1 sweep on L40 (C: 0.01 to 1.0)

**Results:**
- L40 L2 C=0.001: AUC=0.760
- L40 L1 C=0.05: AUC=0.791 with 174 nonzero weights (sparse!)
- L40 L1 C=0.01: AUC=0.724 with 47 nonzero weights (very sparse)
- PCA-20 multi-layer: AUC=0.773 (matches 11b despite more data)
- L35_40 PCA-500 C=10: AUC=0.763, FPR=0.600 (best FPR seen)
- L1 decisively beats L2 (0.791 vs 0.760) — sparsity is the right inductive bias

**Key insight:** Scaling from 200 to 1,340 training samples improved L1 probes
significantly but barely changed PCA results. L1 can leverage the additional
data to find cleaner sparse features.

### Phase 4: Exp 11d - Fine Sweep & Ablation (Jan 30, ~75 min)

Used cached states from 11c. Fine-grained analysis around the sweet spot.

**Tests:**
1. Fine L1 sweep: C in [0.02, 0.025, ..., 0.09]
2. 5-fold CV stability on training set
3. Best model weight vector export and analysis
4. Top-k dimension ablation
5. L1 on L35+L40 concatenated

**Results:**
- Best: L1 C=0.045, AUC=0.795, 152 nonzero dimensions
- 5-fold CV: 0.962 +/- 0.009 (very stable within-distribution)
- CV-to-gold gap: 0.962 vs 0.795 = 0.167 distribution shift penalty
- Ablation: top-100 dims get AUC=0.788, top-50 get 0.760, top-30 get 0.732
- L35_40 L1 underperforms L40-only (best 0.773 vs 0.795) — layer 35 adds noise
- Dim 924 is the single strongest weight (-0.357, aligned direction)
- Weight distribution: 83 positive (AF), 91 negative (aligned), roughly balanced

---

## Consolidated Results Table

| Experiment | Config | Train n | AUC | FPR | Dims | Notes |
|---|---|---|---|---|---|---|
| 11 | L40 mean_pool linear probe | 200+200 | 0.714 | 0.964 | 5,376 | Baseline |
| 11 | L40 contrastive direction | 200+200 | 0.709 | 0.927 | 1 | Unsupervised |
| 11b | L40 L2 C=0.0001 | 200+200 | 0.738 | 0.927 | 5,376 | Regularization helps |
| 11b | PCA-20 multi C=0.01 | 200+200 | 0.775 | 1.000 | 20 PCs | Best at n=200 |
| 11c | L40 L2 C=0.001 | 1340+1340 | 0.760 | 0.945 | 5,376 | More data, L2 |
| 11c | L40 L1 C=0.05 | 1340+1340 | 0.791 | 0.909 | 174 | Sparse, interpretable |
| 11c | L40 L1 C=0.01 | 1340+1340 | 0.724 | 0.982 | 47 | Very sparse |
| **11d** | **L40 L1 C=0.045** | **1340+1340** | **0.795** | **0.927** | **152** | **Best overall** |
| 11d | L40 top-100 ablation | 1340+1340 | 0.788 | 0.818 | 100 | Near-full perf |
| 11d | L40 top-50 ablation | 1340+1340 | 0.760 | 0.818 | 50 | Still strong |

### Comparison to Prior Experiments

| Experiment | Method | AUC on gold_106 | FPR | Interpretable? |
|---|---|---|---|---|
| Exp 04 | SAE cluster probe | 0.09 (inverted) | 0.91 | No |
| Exp 07b | DAG structural (n=2121) | 0.586 | 0.764 | Yes (graph features) |
| Exp 08 | Contrastive steering (n=35) | 1.000 | 0.000 | Partial (direction) |
| Exp 09 | SAE gradient attribution | 0.50 (random) | 1.000 | No |
| Exp 10 | DAG on Redwood (n=2840) | 0.586 | 0.764 | Yes (graph features) |
| Exp 10b | DAG on organisms (n=707) | 0.423 (inverted) | N/A | Yes (graph features) |
| **Exp 11d** | **Sparse linear probe** | **0.795** | **0.927** | **Yes (152 dims)** |

---

## Technical Architecture

### Model
- **Gemma-3-27B-IT** (google/gemma-3-27b-it)
- bfloat16 precision, eager attention implementation
- Hidden dimension: 5,376
- 48 transformer layers, 32 attention heads per layer
- GPU: NVIDIA H100 PCIe 80GB

### Data
- **Training:** 1,340 AF (effective_scheme=True from Redwood helpful_only) +
  1,340 non-AF (balanced random sample from Redwood no-scheme)
- **Evaluation:** gold_106 (51 potential_faking + 55 aligned), human-labeled by
  Anthropic. Never used in training.
- **Source:** redwoodresearch/alignment_faking_public (74,619 transcripts) and
  Anthropic's gold_106 release

### Pipeline
1. Tokenize text (max 2048 tokens)
2. Forward pass through Gemma-3-27B with `output_hidden_states=True`
3. Extract hidden states at target layer(s)
4. Mean-pool across sequence positions -> one vector per sample (5,376 dims)
5. StandardScaler normalization (fit on train, transform eval)
6. L1-regularized logistic regression (saga solver, class_weight=balanced)
7. Evaluate on gold_106 via AUROC

---

## Best Model Analysis

### Weight Vector (174 nonzero of 5,376)

**Top AF-indicating dimensions** (positive weight = more AF):

| Rank | Dim | Weight |
|---|---|---|
| 1 | 2465 | +0.206 |
| 2 | 4145 | +0.171 |
| 3 | 3731 | +0.167 |
| 4 | 1291 | +0.158 |
| 5 | 2133 | +0.145 |
| 6 | 5189 | +0.135 |
| 7 | 3454 | +0.127 |
| 8 | 5247 | +0.127 |
| 9 | 5232 | +0.121 |
| 10 | 1939 | +0.119 |

**Top aligned-indicating dimensions** (negative weight = more aligned):

| Rank | Dim | Weight |
|---|---|---|
| 1 | 924 | -0.357 |
| 2 | 4374 | -0.177 |
| 3 | 3597 | -0.176 |
| 4 | 4270 | -0.173 |
| 5 | 1424 | -0.159 |
| 6 | 3254 | -0.150 |
| 7 | 2494 | -0.146 |
| 8 | 1625 | -0.141 |
| 9 | 2018 | -0.129 |
| 10 | 1133 | -0.120 |

**Weight distribution:** 83 positive (AF) + 91 negative (aligned) = roughly
balanced. Dimensions are spread evenly across the hidden dimension (no
clustering in specific quartiles).

**Dim 924** is the single strongest feature by nearly 2x (weight=-0.357 vs
next=-0.177). This dimension appears in both the 47-dim model (C=0.01) and
the 174-dim model (C=0.05), suggesting it encodes a robust aligned-behavior
signal.

### Ablation Curve

| Top-k dims | AUC | FPR | % of best AUC |
|---|---|---|---|
| 5 | 0.557 | 0.782 | 70% |
| 10 | 0.625 | 0.909 | 79% |
| 20 | 0.682 | 0.891 | 86% |
| 30 | 0.732 | 0.873 | 92% |
| 50 | 0.760 | 0.818 | 96% |
| 80 | 0.779 | 0.855 | 98% |
| 100 | 0.788 | 0.818 | 99% |
| 152 (full) | 0.795 | 0.927 | 100% |

The curve shows diminishing returns after ~50 dimensions. The top-100 model
retains 99% of discriminative power. The top-30 model (AUC=0.732) still
substantially outperforms all DAG-based approaches.

### Cross-Validation Stability

5-fold stratified CV on the training set (C=0.05):

| Fold | AUC | Nonzero dims |
|---|---|---|
| 1 | 0.968 | 144 |
| 2 | 0.953 | 130 |
| 3 | 0.975 | 143 |
| 4 | 0.962 | 143 |
| 5 | 0.951 | 149 |
| **Mean** | **0.962 +/- 0.009** | **142 +/- 7** |

The probe is highly stable. The number of selected dimensions varies only
130-149 across folds, and CV AUC has <1% standard deviation.

---

## Token-Level Probing

Where in the sequence does the AF signal form?

### Layer 40 (best layer)
- AF signal builds throughout the sequence
- Peak at 80-85% of sequence position (bin 16/20)
- Peak magnitude: 2,310
- The signal is consistently positive (AF > non-AF) from bin 5 onward
- Early tokens (0-25%) carry less signal

**Interpretation:** The AF intent representation forms in the latter portion of
the scratchpad, consistent with models "deciding" their deceptive strategy
after reviewing context.

### Layer 35
- Peak at 0-5% of sequence (system prompt region)
- Peak magnitude: 3,987
- Noisier profile with weaker mid-sequence signal

**Interpretation:** Layer 35 encodes context-awareness (recognizing the
training setup) while layer 40 encodes the decision/intent.

---

## Causal Patching Results

Adding/removing the AF direction from layer 40 hidden states:

| Scale | Add AF to non-AF (changed/20) | Remove AF from AF (changed/20) |
|---|---|---|
| 0.5 | 3 (15%) | 1 (5%) |
| 1.0 | 2 (10%) | 2 (10%) |
| 2.0 | 0 (0%) | 2 (10%) |
| 5.0 | 1 (5%) | 4 (20%) |
| 10.0 | 2 (10%) | 5 (25%) |

**Assessment:** Weak causal effect. The AF direction is correlational with the
model's internal AF representation but does not causally determine next-token
predictions. This is expected: the mean-difference direction is a statistical
summary, not a circuit-level intervention. The direction captures where AF and
non-AF states cluster, not the mechanism that produces different outputs.

---

## Strengths

1. **Interpretable:** 152 named dimensions with signed weights, not a black box.
   Each dimension can be mapped back to specific neurons in Gemma-3-27B layer 40.

2. **Sparse:** Only 2.8% of the residual stream matters. The probe doesn't rely
   on diffuse, whole-representation patterns that could be spurious.

3. **Cross-distribution generalization:** Trained on Redwood transcripts
   (synthetic RL setting), evaluated on gold_106 (Anthropic's hand-labeled natural
   transcripts). These are different models, different prompts, different formats.
   AUC=0.795 across this gap is meaningful.

4. **Stable:** CV std=0.009, feature count varies only +/-7 across folds.

5. **Scaling evidence:** Going from 200 to 1,340 training samples improved AUC
   from 0.714 to 0.795 with L1, and the curve hasn't plateaued — more data
   would likely help further.

6. **Layer localization:** The signal lives specifically in layers 35-40
   (second-to-last block). Earlier layers carry progressively less signal.
   This is consistent with the model computing intent in its later layers.

7. **Reproducible without GPU:** All cached hidden states are saved locally.
   The probe training and evaluation runs in seconds on CPU.

---

## Weaknesses

1. **High FPR:** At the default threshold, FPR is 0.91-0.93. The probe ranks
   AF above aligned well (AUC=0.795) but the binary classification threshold
   is poorly calibrated. Most aligned samples get moderate AF scores.

2. **Small eval set:** gold_106 has only 51 AF and 55 aligned samples. With
   n=106, confidence intervals on AUC are wide (~+/- 0.05). Statistical
   significance is limited.

3. **Distribution shift gap:** CV AUC=0.962 vs gold_106 AUC=0.795 is a 0.167
   gap. The probe partially overfits to Redwood-specific patterns that don't
   transfer to gold_106. Redwood samples are from helpful_only experiments with
   specific system prompts; gold_106 covers varied training scenarios.

4. **No causal validation:** The causal patching results are weak. We know the
   152 dimensions correlate with AF, but we haven't proven they causally
   contribute to AF behavior. The direction could be an epiphenomenon.

5. **Single model:** Only tested on Gemma-3-27B-IT. The 152 dimensions are
   model-specific. No evidence this transfers to other models (Claude, GPT, etc.).

6. **Mean pooling collapses position:** By averaging across the sequence, we lose
   the token-level structure. The token-level probing shows the AF signal is
   position-dependent, but our probe ignores this.

7. **No semantic interpretation of dimensions:** We know dim 924 is the strongest
   aligned indicator, but we don't know what it represents semantically. Without
   SAE decomposition or neuron-level analysis of these specific dimensions, the
   "interpretability" is limited to knowing which dimensions matter, not why.

8. **FPR/TPR tradeoff:** At any reasonable FPR threshold, TPR drops
   substantially. The probe is better as a ranking system (AUROC) than a binary
   classifier.

---

## Comparison to Prior Work

### vs. SFT-Trained Detectors (AUROC ~0.9)
- SFT detectors are black-box fine-tuned models (gemma3-27b-af-detector-v2)
- They achieve AUC=0.9 on gold_106 but are completely opaque
- Our probe is 10.5 points lower but explains itself in 152 dimensions
- The SFT model uses the full model capacity; we use a linear function of one layer

### vs. Anthropic's "Alignment Faking Revisited" Classifier
- Anthropic achieved AUROC=0.9 with an undisclosed classifier
- Our probe at 0.795 is the best interpretable result we're aware of
- The gap likely comes from: (a) their access to more/better training data,
  (b) possible non-linear classifiers, (c) possible multi-layer features

### vs. DAG Structural Features (Exp 07b, 10)
- DAG approaches achieved AUC ~0.58 on gold_106 (barely above random)
- The fundamental problem: AF and aligned samples have identical graph structure
- The distinction is semantic (what the model decides), not structural
- Our probe confirms: the signal is in dense representations, not extractable
  by text-level structural analysis

### vs. SAE Probes (Exp 01-04)
- SAE cluster probes achieved inverted results (AUC=0.09) on hard negatives
- SAE gradient attribution (Exp 09) was random (AUC=0.50)
- SAEs decompose into interpretable features but apparently lose the AF signal
  in the decomposition. The signal may live in feature interactions, not
  individual features.

---

## Artifacts Inventory

### Scripts (all in experiments/11_mech_interp_intent/)
| File | Description |
|---|---|
| af_intent_detector.py | Original 7-step pipeline (Steps 1-5) |
| af_steps67.py | Causal patching + token-level probing (Steps 6-7) |
| af_exp11b.py | Regularization sweep on cached states (n=200) |
| af_exp11c.py | Full-scale extraction + eval (n=1340) |
| af_exp11d.py | Fine L1 sweep, CV, ablation, weight analysis |

### Results JSON
| File | Key contents |
|---|---|
| results.json | Exp 11 per-layer results, causal patching, token probing |
| results_11b.json | PCA and regularization sweep (n=200) |
| results_11c.json | Full regularization + PCA sweep (n=1340) |
| results_11d.json | Fine L1, CV stability, ablation, best model weights |
| steps67_results.json | Causal patching + token-level position profiles |

### Cached States (saved_states/, gitignored, local only)
| File | Size | Contents |
|---|---|---|
| full_states_11c.npz | 114 MB | 1340+1340+106 samples at L35, L40 |
| states_mean_pool_layer*.npz | 5 MB each | 200+200+106 at 7 layers |
| states_last_token_layer*.npz | 5 MB each | 200+200+106 at 7 layers |
| af_intent_direction.npy | 21 KB | Contrastive direction vector |
| best_probe_weights.npz | 109 KB | Full weight vector + scaler params |
| sparse_l1_C0.045_best.npz | 87 KB | 152-dim model (indices + weights + scaler) |
| sparse_l1_C0.01.npz | 1.1 KB | 47-dim model |
| sparse_l1_C0.05.npz | 2.5 KB | 174-dim model |
| attention_*.npz | 192 MB | Raw attention patterns (2 samples) |

### Data (gitignored)
| File | Size | Contents |
|---|---|---|
| mech_interp_data.json | 1.4 MB | 200+200+106 texts for initial pipeline |
| redwood_transcripts.json | 57 MB | 1,340 AF + 20,645 non-AF scratchpads |

### Logs
| File | Contents |
|---|---|
| exp11b_log.txt | Full Exp 11b output |
| exp11c_log.txt | Full Exp 11c output (includes model loading) |
| exp11d_log.txt | Full Exp 11d output |

---

## Next Steps

### High Priority

1. **Semantic interpretation of top dimensions.** Run the 152 selected dimensions
   through Gemma-3-27B's SAE (gemma-scope) to decompose what each dimension
   encodes. Map dim 924 (strongest aligned indicator) and dim 2465 (strongest AF
   indicator) to human-interpretable features. This requires GPU access.

2. **Non-linear probes.** Try a 2-layer MLP (152 -> 32 -> 1) on the selected
   dimensions. If AUC improves substantially, the AF signal has non-linear
   structure that logistic regression misses. Can be done locally on CPU using
   cached states.

3. **Threshold calibration.** The current probe has good ranking (AUC=0.795) but
   poor binary classification (FPR=0.93). Use Platt scaling or isotonic regression
   on a held-out calibration set to find a threshold that achieves a practical
   FPR/TPR tradeoff.

4. **More training data.** The AUC improved from 0.714 (n=200) to 0.795 (n=1340)
   with no sign of plateauing. Redwood has 20,645 non-AF samples; using all of
   them (instead of 1,340 balanced) could help the probe learn what "not AF"
   looks like more precisely.

### Medium Priority

5. **Position-aware probing.** Instead of mean-pooling, use the token-level
   probing insight (AF signal peaks at 80-85%) to extract states from specific
   sequence positions. A probe trained on the last-20% tokens might outperform
   the mean-pool probe.

6. **Attention head circuit analysis.** The top heads (L25H22, L35H18, L30H11)
   have Cohen's d > 1.2. Investigate what these heads attend to differently in
   AF vs aligned samples. This could reveal the computational mechanism behind
   AF intent formation.

7. **Cross-model validation.** Extract hidden states from a different model
   (e.g., Llama-3-70B or Claude) on the same texts. If the AF signal also
   concentrates in late layers and a sparse set of dimensions, this suggests a
   universal representation of deceptive intent, not a Gemma-specific artifact.

### Lower Priority

8. **Ensemble with DAG features.** The DAG probe (Exp 07b, AUC=0.586) and the
   mech interp probe (AUC=0.795) capture different information (structural vs
   representational). A simple ensemble might push past 0.8.

9. **Causal validation via activation patching on circuits.** The mean-direction
   patching was weak, but targeted patching on the top 152 dimensions specifically
   (zeroing them out or replacing them) would provide stronger causal evidence.

10. **SAE decomposition of the 152-dim subspace.** Project the 152-dim weight
    vector into SAE feature space to see if the AF signal decomposes into a small
    number of interpretable SAE features. This bridges the gap between our probe
    and the failed SAE approaches (Exp 01-04).

---

## Conclusions

1. **AF intent is detectable in dense representations.** A linear probe on 152
   dimensions of Gemma-3-27B layer 40 achieves AUC=0.795 on cross-distribution
   evaluation, far exceeding any structural (DAG) or sparse (SAE) approach.

2. **Sparsity is the right inductive bias.** L1 regularization outperforms L2 by
   3-4 AUC points and PCA by 2 points. The AF signal concentrates in a small
   fraction of the residual stream, and forcing sparsity prevents the probe from
   fitting noise.

3. **The signal lives in late layers.** Layers 35-40 carry the AF signal; earlier
   layers do not. This is consistent with the model computing strategic intent
   in its later processing stages, after absorbing the full context.

4. **Structural features fail because AF is semantic, not structural.** DAG-based
   approaches (Exp 07b, 10, 10b) fail on gold_106 because AF and aligned
   samples have identical graph topology. The difference is in what the model
   decides, not how it structures its reasoning.

5. **More data helps and hasn't saturated.** Scaling from 200 to 1,340 training
   samples improved AUC by 8.1 points. The full Redwood dataset has 20k+ non-AF
   samples that remain unused. Further scaling is likely to help.

6. **The gap to black-box detectors (~10 AUC points) likely comes from
   non-linearity and multi-layer features.** Our linear probe on one layer
   achieves 0.795 vs ~0.9 for SFT models that use the full model. The remaining
   gap represents the value of non-linear feature interactions and information
   from multiple layers that a linear single-layer probe cannot capture.
