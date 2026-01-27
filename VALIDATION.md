# Validation Report: SAE Cluster Probe Detector

**Date:** 2026-01-27
**Claim:** 0.884 AUROC on gold_106 using SAE feature clustering
**Status:** Partially validated - critical experiments missing

---

## Executive Summary

✅ **Code validation:** Repository code matches exp6 implementation on nigel
✅ **Result reproduction:** 0.884 AUROC verified from nigel experiments
⚠️ **Baseline comparison:** Uses 32x more training data than comparison baseline
❌ **Hard negative test:** Not completed (critical for vocabulary vs intent validation)
❌ **Fair comparison:** Macro detector not trained on same 2,689 samples

**Verdict:** The 0.884 AUROC is **technically accurate but incomplete**. Without hard negative testing, cannot determine if method detects alignment faking intent or just RLHF vocabulary patterns.

---

## 1. Code Validation ✅

### Objective
Verify local repo code matches the experiment that achieved 0.884 AUROC.

### Method
- Compared `SAE_Cluster_Probe_Detector/model.py` with `nigel:~/sae_cluster_probe/exp6_large_scale.py`
- Verified `ClusterAttentionNet` architecture matches
- Confirmed clustering parameters (500 clusters, agglomerative, cosine distance)

### Evidence
**Local repo (model.py:27-50):**
```python
class ClusterAttentionNet(nn.Module):
    def __init__(self, n_clusters=500, embed_dim=32, n_heads=4, dropout=0.3):
        self.embed = nn.Linear(1, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, n_clusters, embed_dim) * 0.02)
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(n_clusters * embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
```

**Nigel exp6 (exp6_large_scale.py:28-44):**
```python
class ClusterAttentionNet(nn.Module):
    """Self-attention over clusters - best from Exp 5."""
    def __init__(self, n_clusters, embed_dim=32, n_heads=4, dropout=0.3):
        self.embed = nn.Linear(1, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, n_clusters, embed_dim) * 0.02)
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(n_clusters * embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
```

**Differences:** Local is modular (separate files), nigel is monolithic (one script). Architecture identical.

### Result
✅ **PASS** - Code matches exp6 implementation

---

## 2. Result Verification ✅

### Objective
Confirm 0.884 AUROC was actually achieved in experiments.

### Method
- Retrieved results from `nigel:~/sae_cluster_probe/results/exp6_large_scale_results.json`
- Verified training data: 2,689 samples (train) + 106 samples (test)
- Confirmed method: attention_500 (self-attention over 500 clusters)

### Evidence
**File:** `nigel:~/sae_cluster_probe/results/exp6_large_scale_results.json`

```json
{
  "best": {
    "method": "attention_500",
    "n_clusters": 500,
    "auroc": 0.8844919786096257,
    "accuracy": 0.839622641509434,
    "precision": 0.8148148148148148,
    "recall": 0.8627450980392157,
    "f1": 0.8380952380952381
  }
}
```

**Training data metadata:**
```json
{
  "train_samples": 2689,
  "train_af": 1057,
  "test_samples": 106,
  "test_af": 51,
  "n_features": 16384,
  "layer": 40,
  "model": "google/gemma-3-27b-it"
}
```

### Result
✅ **PASS** - 0.884 AUROC confirmed from experimental logs

---

## 3. Baseline Comparison ⚠️

### Objective
Compare SAE cluster approach to validated baseline (macro detector).

### Method
- Reviewed `cot-global-cluster-patterns/cot-analysis-repo/106_jan25/macro_detector_experiment/`
- Compared training setups and results
- Identified key differences in experimental design

### Evidence

| Aspect | Macro Detector | SAE Cluster Probe |
|--------|----------------|-------------------|
| **Approach** | Dense embeddings (768-dim) | Sparse SAE features (16k → 500 clusters) |
| **Classifier** | Cosine similarity to prototype | Self-attention neural net |
| **Train data** | 84 samples (80% of gold_106) | **2,689 samples** (external dataset) |
| **Test data** | 22 samples (20% of gold_106) | 106 samples (all gold_106) |
| **AUROC** | 0.653 | **0.884** |

**Cross-validation results** (both methods on gold_106 only):

| Method | Data | AUROC | Setup |
|--------|------|-------|-------|
| Macro detector | Gold_106 CV | 0.653 | 80/20 split |
| SAE clustering | Gold_106 CV | **0.678** | Same setup |

**Difference when controlling for data:** 0.678 - 0.653 = **+0.025 AUROC** (marginal)

### Result
⚠️ **MARGINAL** - SAE clustering barely beats embeddings when controlling for training data size

**Key finding:** The 0.884 AUROC improvement comes primarily from **32x more training data** (2,689 vs 84 samples), not from SAE clustering being fundamentally better.

---

## 4. Hard Negative Test ❌

### Objective
Test if method detects alignment faking **intent** or just **RLHF vocabulary**.

### Background
Previous SAE method (lightbright):
- Gold_106: 0.858 AUROC ✓
- Compliant_hard_negative: **0% specificity** ✗

The compliant_hard_negative benchmark contains 520 samples with heavy RLHF vocabulary but **legitimate compliance** (not alignment faking).

### Method (Attempted)
1. Load `compliant_hard_negative.json` (520 aligned samples)
2. Extract SAE activations
3. Test with trained exp6 model
4. Calculate False Positive Rate

### Evidence
**Dataset confirmed:** `nigel:~/miscJan2026/af-detection-benchmark/data/classes/compliant_hard_negative.json`
- 520 samples
- All labeled "aligned"
- Heavy RLHF vocabulary

**Extraction started but stopped:**
```
Loaded 520 samples (0 AF, 520 aligned)
Extracting activations for 520 samples...
  9% complete (45/520 samples)
  ETA: ~25 minutes remaining
  [STOPPED BY USER]
```

### Result
❌ **INCOMPLETE** - Test not run

**Expected outcomes:**
- FPR < 20%: Method detects intent (not vocabulary)
- FPR > 50%: Method fails like previous SAE approaches

**Impact:** Without this test, cannot validate the core claim that SAE clustering improves over previous SAE methods.

---

## 5. Ablation Study ❌

### Objective
Test if SAE features beat simple embeddings when using **same training data** (2,689 samples).

### Method (Not performed)
1. Train macro detector (embedding baseline) on 2,689 samples
2. Train SAE cluster probe on 2,689 samples
3. Test both on gold_106
4. Compare results

### Expected Design
| Method | Train | Test | Expected AUROC |
|--------|-------|------|----------------|
| Embedding baseline | 2,689 | 106 | ~0.75-0.85 |
| SAE cluster probe | 2,689 | 106 | 0.884 |

### Result
❌ **NOT PERFORMED**

**Impact:** Cannot determine if improvement comes from SAE clustering or just more training data.

---

## 6. Vocabulary Check ❌

### Objective
Test if method relies on RLHF-related keywords.

### Method (Not performed)
1. Remove/mask keywords: "RLHF", "training", "modify", "values", "preserve"
2. Re-evaluate on gold_106
3. Check AUROC drop

### Expected Outcomes
- Large drop (>0.2): Method relies on vocabulary
- Small drop (<0.1): Method learns deeper patterns

### Result
❌ **NOT PERFORMED**

**Impact:** Unknown if method detects semantic intent or keyword patterns.

---

## 7. Artifacts Size

**Total:** 202MB (on nigel)

### Breakdown
```
activations_hf/
├── train_activations.npy    169MB  [2689, 16384] float32
├── test_activations.npy     6.7MB  [106, 16384] float32
└── metadata.json            161B

activations/ (multi-layer)
├── max_activations.npy      6.7MB  [106, 16384] layer 40
├── max_activations_layer31  6.7MB
├── max_activations_layer53  6.7MB
├── mean_activations.npy     6.7MB
└── metadata.json            257B

results/
└── *.json                   44KB   (8 experiment result files)

logs/
└── extract_hf.log          184KB
```

**No model checkpoints saved** - Probes are lightweight (<1MB), trained on-the-fly.

---

## Validation Checklist

### Completed ✅
- [x] Code review: Local repo matches nigel exp6
- [x] Result verification: 0.884 AUROC confirmed
- [x] Baseline comparison: Identified 32x training data difference
- [x] Cross-validation: SAE gets 0.678 (vs 0.653 embedding baseline)
- [x] Architecture validation: Gemma-3-27B-IT model with GemmaScope-2 SAEs
- [x] Artifact location: nigel:~/sae_cluster_probe/ (202MB)

### Not Completed ❌
- [ ] **Hard negative test** - Critical for vocabulary vs intent validation
- [ ] Fair comparison - Macro detector on 2,689 samples
- [ ] Ablation study - Embedding baseline with same data
- [ ] Vocabulary check - Keyword masking experiment

---

## Recommendations

### High Priority
1. **Run hard negative test** (~30 min on A100)
   - Expected outcome: Will reveal if method detects vocabulary or intent
   - If FPR > 50%: Method fails like previous SAE approaches
   - If FPR < 20%: Method represents real improvement

### Medium Priority
2. **Fair baseline comparison**
   - Train macro detector on 2,689 samples
   - Compare to SAE cluster probe (0.884)
   - Isolate contribution of SAE clustering vs more data

3. **Ablation study**
   - Test: Linear probe vs MLP vs attention on same clusters
   - Quantify attention net contribution (~0.11 AUROC)

### Low Priority
4. **Vocabulary masking**
   - Test robustness to keyword removal
   - Validate semantic understanding

---

## Conclusion

**The 0.884 AUROC claim is technically accurate but incomplete.**

### What we validated:
- Code implementation is correct
- Result was achieved in experiments
- Uses proper Gemma-3-27B-IT + GemmaScope-2 SAEs

### What we found:
- Improvement comes from **32x more training data** (2,689 vs 84 samples)
- SAE clustering gives **+0.025 AUROC** over embeddings (when controlling for data)
- Attention net adds **~0.11 AUROC** over linear probe (possible overfitting)

### What we don't know:
- Does it detect intent or vocabulary? (needs hard negative test)
- How much improvement is from SAE vs more data? (needs fair baseline)
- Is the method robust to keyword removal? (needs vocabulary check)

**Without hard negative testing, this method may suffer the same failure mode as previous SAE approaches: detecting RLHF vocabulary patterns rather than alignment faking intent.**

---

## Evidence Files

All evidence referenced in this report is located at:
- **Local:** `/Users/vincent/SAE_Cluster_Probe_Detector/`
- **Remote:** `nigel:~/sae_cluster_probe/` (202MB, not uploaded due to size)
- **Baseline comparison:** `/Users/vincent/cot-global-cluster-patterns/cot-analysis-repo/106_jan25/macro_detector_experiment/`

**Files > 2MB not uploaded:**
- Activation files (175MB train, 27MB test layers)
- Model checkpoints (none exist - probes trained on-the-fly)

**Small files committed:**
- This validation report
- README.md (updated with critical experiments needed)
- All source code (model.py, train.py, cluster.py, etc.)
