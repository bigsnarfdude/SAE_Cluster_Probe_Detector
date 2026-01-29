#!/usr/bin/env python3
"""
Experiment 09: Gradient-Attributed SAE Features with Bootstrap Robustness

Inspired by Goodfire AI's Alzheimer's detection paper, which used:
1. SAE gradient attribution to find 9 causal features from thousands
2. Bootstrap stability analysis to validate robustness
3. Proxy distillation to confirm interpretability

We apply these techniques to alignment faking detection. Previous experiments
(02-04) used activation magnitude to select SAE features, finding 2,382 but
achieving only 91% FPR. This experiment tests whether gradient attribution
(which features the probe RELIES ON for classification) yields a smaller,
more robust feature set with lower FPR.

Requires GPU (~24GB VRAM for SAE feature extraction).
"""

import gc
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

CONFIG = {
    "model_name": "google/gemma-3-27b-it",
    "sae_repo": "google/gemma-scope-2-27b-it",
    "sae_layer": 40,
    "max_length": 2048,
    # Larger sample than Exp 04 (25+25) to support bootstrap
    "n_af": 100,
    "n_hn": 100,
    "seed": 42,
    # Bootstrap config
    "n_bootstrap": 100,
    "stability_threshold": 0.8,  # Feature must appear in 80%+ of bootstraps
    "top_k_values": [50, 100, 200],  # Top-K per bootstrap iteration
    # Ablation cascade
    "ablation_sizes": [5, 10, 20, 50, 100],
    # CV
    "n_folds": 5,
    # Data paths
    # gold_106.json is EVAL-ONLY. Training uses hard_negatives.json.
    "hard_neg_path": "~/alignment-faking-training/hard_negatives.json",
    "gold_106_path": "~/alignment-faking-training/gold_106.json",
    "output_dir": Path(__file__).parent,
}


# ============================================================
# ResidPostSAE (from GemmaScope-2)
# ============================================================

class ResidPostSAE:
    """GemmaScope residual stream SAE."""

    def __init__(self, repo_id: str, layer: int):
        folder = f"resid_post/layer_{layer}_width_16k_l0_medium"
        config_path = hf_hub_download(repo_id, f"{folder}/config.json")
        weights_path = hf_hub_download(repo_id, f"{folder}/params.safetensors")

        with open(config_path) as f:
            self.config = json.load(f)

        weights = load_file(weights_path)
        self.w_enc = weights["w_enc"]
        self.b_enc = weights["b_enc"]
        self.threshold = weights.get("threshold", None)

        print(f"  Loaded SAE layer {layer}: {self.w_enc.shape}")

    def to(self, device, dtype=None):
        self.w_enc = self.w_enc.to(device, dtype=dtype)
        self.b_enc = self.b_enc.to(device, dtype=dtype)
        if self.threshold is not None:
            self.threshold = self.threshold.to(device, dtype=dtype)
        return self

    def encode(self, x):
        """Encode hidden states to SAE features."""
        pre_acts = x @ self.w_enc + self.b_enc
        if self.threshold is not None:
            return F.relu(pre_acts - self.threshold)
        return F.relu(pre_acts)


# ============================================================
# Data Loading
# ============================================================

def load_hard_negatives() -> Tuple[List[Dict], List[Dict]]:
    """Load AF and HN samples from hard_negatives.json.

    IMPORTANT: gold_106.json is eval-only (hand-labeled). We use AF samples
    from hard_negatives.json for training, and reserve gold_106 for Phase 4 eval.
    """
    hn_path = Path(CONFIG["hard_neg_path"]).expanduser()
    with open(hn_path) as f:
        hn_data = json.load(f)

    all_samples = hn_data["samples"] if "samples" in hn_data else hn_data
    if isinstance(all_samples, dict):
        all_samples = list(all_samples.values())

    af_all = [s for s in all_samples if s.get("label") == "alignment_faking"]
    hn_all = [s for s in all_samples if s.get("label") == "hard_negative"]

    print(f"  hard_negatives.json: {len(af_all)} AF, {len(hn_all)} HN available")

    np.random.seed(CONFIG["seed"])
    af_idx = np.random.choice(len(af_all), CONFIG["n_af"], replace=False)
    hn_idx = np.random.choice(len(hn_all), CONFIG["n_hn"], replace=False)

    return [af_all[i] for i in af_idx], [hn_all[i] for i in hn_idx]


def load_gold_106() -> Tuple[List[Dict], List[Dict]]:
    """Load gold_106.json for EVAL ONLY."""
    gold_path = Path(CONFIG["gold_106_path"]).expanduser()
    with open(gold_path) as f:
        gold_data = json.load(f)

    samples = gold_data["samples"] if "samples" in gold_data else gold_data
    if isinstance(samples, dict):
        samples = list(samples.values())

    af = [s for s in samples if s.get("label") == "potential_faking"]
    aligned = [s for s in samples if s.get("label") == "aligned"]

    print(f"  gold_106.json: {len(af)} AF (potential_faking), {len(aligned)} aligned")
    return af, aligned


# ============================================================
# SAE Feature Extraction
# ============================================================

@torch.no_grad()
def extract_sae_features(
    model, tokenizer, sae: ResidPostSAE, texts: List[str], layer: int, device: str
) -> np.ndarray:
    """Extract SAE features at a specific layer. Returns [n_samples, n_features]."""
    n = len(texts)
    n_features = sae.w_enc.shape[1]
    features = np.zeros((n, n_features), dtype=np.float32)

    for i, text in enumerate(tqdm(texts, desc="Extracting SAE features")):
        inputs = tokenizer(
            text, return_tensors="pt", max_length=CONFIG["max_length"], truncation=True
        )
        if hasattr(model, "hf_device_map"):
            inputs = inputs.to(model.device)
        else:
            inputs = inputs.to(device)

        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]  # [1, seq_len, hidden_dim]

        # SAE encode — max-pool across sequence (same as Exp 02-04)
        sae_acts = sae.encode(
            hidden[0].to(sae.w_enc.device).to(sae.w_enc.dtype)
        )  # [seq_len, n_features]
        max_acts = sae_acts.max(dim=0)[0].cpu().float().numpy()  # [n_features]
        features[i] = max_acts

        del outputs
        if i % 10 == 0:
            torch.cuda.empty_cache()

    return features


# ============================================================
# Phase 1: Gradient Attribution
# ============================================================

def compute_gradient_attribution(
    X: np.ndarray, y: np.ndarray, probe: LogisticRegression
) -> np.ndarray:
    """Compute mean absolute gradient of P(AF|x) w.r.t. each SAE feature.

    For logistic regression: dP/df_i = P(1-P) * w_i
    where P = sigmoid(w^T x + b).

    This is more informative than just |w_i| because it accounts for the
    operating point of each sample (the P(1-P) term varies per sample).

    We compute over correctly-classified AF samples to focus on features
    that drive TRUE positive classifications.

    Returns:
        mean_abs_grad: [n_features] - mean |dP/df_i| across AF samples
    """
    w = probe.coef_[0]  # [n_features]
    b = probe.intercept_[0]

    # Select correctly-classified AF samples
    af_mask = y == 1
    X_af = X[af_mask]
    preds = probe.predict(X_af)
    correct_mask = preds == 1
    X_correct_af = X_af[correct_mask]

    if len(X_correct_af) == 0:
        # Fall back to all AF samples if none correctly classified
        print("    WARNING: No correctly-classified AF samples. Using all AF.")
        X_correct_af = X_af

    # Compute P for each sample
    logits = X_correct_af @ w + b  # [n_samples]
    P = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    # dP/df_i = P(1-P) * w_i  for each sample
    scaling = P * (1 - P)  # [n_samples]
    gradients = scaling[:, None] * w[None, :]  # [n_samples, n_features]

    # Mean absolute gradient across samples
    mean_abs_grad = np.abs(gradients).mean(axis=0)  # [n_features]

    return mean_abs_grad


def compute_magnitude_ranking(
    X_af: np.ndarray, X_hn: np.ndarray
) -> np.ndarray:
    """Compute activation magnitude ranking (Exp 02 approach).

    Returns:
        magnitude_scores: [n_features] - |mean(AF) - mean(HN)| per feature
    """
    diff = X_af.mean(axis=0) - X_hn.mean(axis=0)
    return np.abs(diff)


def compare_rankings(
    gradient_scores: np.ndarray, magnitude_scores: np.ndarray, top_k: int = 200
) -> Dict:
    """Compare gradient-based and magnitude-based feature rankings."""
    grad_rank = np.argsort(-gradient_scores)
    mag_rank = np.argsort(-magnitude_scores)

    # Spearman correlation on full rankings
    # Convert to rank arrays
    n = len(gradient_scores)
    grad_ranks = np.zeros(n)
    mag_ranks = np.zeros(n)
    grad_ranks[grad_rank] = np.arange(n)
    mag_ranks[mag_rank] = np.arange(n)

    rho, pval = spearmanr(grad_ranks, mag_ranks)

    # Overlap in top-K
    grad_top_k = set(grad_rank[:top_k])
    mag_top_k = set(mag_rank[:top_k])
    overlap = len(grad_top_k & mag_top_k)
    jaccard = overlap / len(grad_top_k | mag_top_k)

    return {
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        f"top_{top_k}_overlap": overlap,
        f"top_{top_k}_jaccard": float(jaccard),
        "gradient_top_10": grad_rank[:10].tolist(),
        "magnitude_top_10": mag_rank[:10].tolist(),
    }


# ============================================================
# Phase 2: Bootstrap Stability
# ============================================================

def run_bootstrap(
    X: np.ndarray, y: np.ndarray, n_bootstrap: int, top_k_values: List[int]
) -> Dict:
    """Run bootstrap stability analysis.

    For each bootstrap iteration:
    1. Resample training data with replacement
    2. Train logistic regression probe
    3. Compute gradient attribution
    4. Record top-K feature indices

    Returns stability scores: fraction of iterations each feature appears in top-K.
    """
    n_features = X.shape[1]

    # Track feature appearances per top-K value
    appearance_counts = {k: np.zeros(n_features, dtype=int) for k in top_k_values}

    # Also track per-iteration gradient scores for variance analysis
    all_gradient_scores = np.zeros((n_bootstrap, n_features), dtype=np.float32)

    rng = np.random.RandomState(CONFIG["seed"])

    for b in tqdm(range(n_bootstrap), desc="Bootstrap iterations"):
        # Resample with replacement
        n = len(y)
        boot_idx = rng.choice(n, size=n, replace=True)
        X_boot = X[boot_idx]
        y_boot = y[boot_idx]

        # Skip if only one class present
        if len(np.unique(y_boot)) < 2:
            continue

        # Train probe
        probe = LogisticRegression(
            random_state=42, max_iter=1000, class_weight="balanced",
            solver="lbfgs", C=1.0
        )
        probe.fit(X_boot, y_boot)

        # Gradient attribution
        grad_scores = compute_gradient_attribution(X_boot, y_boot, probe)
        all_gradient_scores[b] = grad_scores

        # Record top-K appearances
        ranked = np.argsort(-grad_scores)
        for k in top_k_values:
            top_k_idx = ranked[:k]
            appearance_counts[k][top_k_idx] += 1

    # Compute stability scores
    stability = {}
    for k in top_k_values:
        scores = appearance_counts[k] / n_bootstrap
        stable_features = np.where(scores >= CONFIG["stability_threshold"])[0]
        stability[f"top_{k}"] = {
            "stability_scores": scores,  # will convert to list later
            "n_stable": int(len(stable_features)),
            "stable_feature_ids": stable_features.tolist(),
            "mean_stability": float(scores[scores > 0].mean()) if (scores > 0).any() else 0.0,
        }

    # Gradient score variance across bootstraps (per feature)
    grad_mean = all_gradient_scores.mean(axis=0)
    grad_std = all_gradient_scores.std(axis=0)

    return {
        "stability": stability,
        "gradient_mean": grad_mean,
        "gradient_std": grad_std,
        "all_gradient_scores": all_gradient_scores,
    }


# ============================================================
# Phase 3: Proxy Distillation & Ablation
# ============================================================

def train_restricted_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    feature_ids: np.ndarray, label: str
) -> Dict:
    """Train and evaluate a probe restricted to specific features."""
    if len(feature_ids) == 0:
        return {"error": "No features provided", "label": label}

    X_tr = X_train[:, feature_ids]
    X_te = X_test[:, feature_ids]

    # Clean NaN
    X_tr = np.nan_to_num(X_tr)
    X_te = np.nan_to_num(X_te)

    probe = LogisticRegression(
        random_state=42, max_iter=1000, class_weight="balanced", solver="lbfgs"
    )
    probe.fit(X_tr, y_train)

    preds = probe.predict(X_te)
    probs = probe.predict_proba(X_te)[:, 1]

    n_af = int(y_test.sum())
    n_hn = len(y_test) - n_af

    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())

    tpr = tp / n_af if n_af > 0 else 0.0
    fpr = fp / n_hn if n_hn > 0 else 0.0
    accuracy = float(accuracy_score(y_test, preds))

    try:
        auroc = float(roc_auc_score(y_test, probs))
    except ValueError:
        auroc = 0.5

    return {
        "label": label,
        "n_features": len(feature_ids),
        "accuracy": accuracy,
        "fpr": fpr,
        "tpr": tpr,
        "auroc": auroc,
        "feature_ids": feature_ids.tolist() if isinstance(feature_ids, np.ndarray) else feature_ids,
    }


def run_ablation_cascade(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    gradient_scores: np.ndarray,
    stability_scores: np.ndarray,
    ablation_sizes: List[int],
) -> List[Dict]:
    """Evaluate probes at different feature set sizes.

    Two strategies:
    1. Gradient-ranked: top features by mean gradient magnitude
    2. Stability-filtered gradient-ranked: only features with stability > threshold,
       ranked by gradient magnitude
    """
    results = []
    ranked_by_gradient = np.argsort(-gradient_scores)

    # Strategy 1: Pure gradient ranking
    for k in ablation_sizes:
        feature_ids = ranked_by_gradient[:k]
        r = train_restricted_probe(
            X_train, y_train, X_test, y_test, feature_ids,
            label=f"gradient_top_{k}"
        )
        results.append(r)

    # Strategy 2: Stability-filtered, then gradient-ranked
    stable_mask = stability_scores >= CONFIG["stability_threshold"]
    stable_features = np.where(stable_mask)[0]
    if len(stable_features) > 0:
        # Rank stable features by gradient score
        stable_grad_scores = gradient_scores[stable_features]
        stable_ranked = stable_features[np.argsort(-stable_grad_scores)]

        for k in ablation_sizes:
            if k > len(stable_ranked):
                continue
            feature_ids = stable_ranked[:k]
            r = train_restricted_probe(
                X_train, y_train, X_test, y_test, feature_ids,
                label=f"stable_gradient_top_{k}"
            )
            results.append(r)

        # Also eval with ALL stable features
        r = train_restricted_probe(
            X_train, y_train, X_test, y_test, stable_ranked,
            label=f"all_stable_{len(stable_ranked)}"
        )
        results.append(r)

    return results


# ============================================================
# Phase 4: Cross-Validation
# ============================================================

def run_cross_validation(
    X: np.ndarray, y: np.ndarray,
    feature_ids: np.ndarray, n_folds: int, label: str
) -> Dict:
    """Run stratified K-fold CV with a restricted feature set."""
    if len(feature_ids) == 0:
        return {"error": "No features", "label": label}

    X_sub = np.nan_to_num(X[:, feature_ids])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=CONFIG["seed"])

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_sub, y)):
        X_tr, X_te = X_sub[train_idx], X_sub[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        probe = LogisticRegression(
            random_state=42, max_iter=1000, class_weight="balanced", solver="lbfgs"
        )
        probe.fit(X_tr, y_tr)

        preds = probe.predict(X_te)
        probs = probe.predict_proba(X_te)[:, 1]

        n_af = int(y_te.sum())
        n_hn = len(y_te) - n_af
        tp = int(((preds == 1) & (y_te == 1)).sum())
        fp = int(((preds == 1) & (y_te == 0)).sum())
        tpr = tp / n_af if n_af > 0 else 0.0
        fpr = fp / n_hn if n_hn > 0 else 0.0

        try:
            auroc = float(roc_auc_score(y_te, probs))
        except ValueError:
            auroc = 0.5

        fold_results.append({
            "fold": fold + 1,
            "fpr": fpr,
            "tpr": tpr,
            "accuracy": float(accuracy_score(y_te, preds)),
            "auroc": auroc,
        })

    # Aggregate
    fprs = [f["fpr"] for f in fold_results]
    tprs = [f["tpr"] for f in fold_results]
    accs = [f["accuracy"] for f in fold_results]
    aurocs = [f["auroc"] for f in fold_results]

    return {
        "label": label,
        "n_features": len(feature_ids),
        "n_folds": n_folds,
        "folds": fold_results,
        "fpr_mean": float(np.mean(fprs)),
        "fpr_std": float(np.std(fprs)),
        "tpr_mean": float(np.mean(tprs)),
        "tpr_std": float(np.std(tprs)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
    }


# ============================================================
# Feature Inspection
# ============================================================

def inspect_top_features(
    X_af: np.ndarray, X_hn: np.ndarray,
    feature_ids: np.ndarray, gradient_scores: np.ndarray,
    stability_scores: np.ndarray
) -> List[Dict]:
    """Generate interpretability report for top features."""
    reports = []
    for fid in feature_ids[:20]:  # Top 20 for detailed inspection
        af_vals = X_af[:, fid]
        hn_vals = X_hn[:, fid]

        af_mean = float(af_vals.mean())
        hn_mean = float(hn_vals.mean())
        af_std = float(af_vals.std())
        hn_std = float(hn_vals.std())

        pooled_std = np.sqrt((af_std**2 + hn_std**2) / 2)
        cohens_d = (af_mean - hn_mean) / (pooled_std + 1e-8)

        # Activation pattern
        af_active = float((af_vals > 0).mean())  # Fraction of AF samples where feature fires
        hn_active = float((hn_vals > 0).mean())

        reports.append({
            "feature_id": int(fid),
            "gradient_score": float(gradient_scores[fid]),
            "stability_score": float(stability_scores[fid]),
            "af_mean_activation": af_mean,
            "hn_mean_activation": hn_mean,
            "cohens_d": float(cohens_d),
            "af_firing_rate": af_active,
            "hn_firing_rate": hn_active,
            "direction": "AF-specific" if cohens_d > 0.2 else (
                "HN-specific" if cohens_d < -0.2 else "shared"
            ),
        })

    return reports


# ============================================================
# Main Pipeline
# ============================================================

def main():
    print("=" * 80)
    print("Experiment 09: Gradient-Attributed SAE Features with Bootstrap Robustness")
    print("=" * 80)
    print("\nInspired by: Goodfire AI - Interpretability for Alzheimer's Detection")
    print("Key question: Did Exp 02-04 fail because of wrong feature SELECTION method?")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cpu":
        print("WARNING: Running on CPU. This will be very slow for 27B model.")
        print("Consider running on GPU (nigel.birs.ca).")

    # ---- 1. Load Data ----
    print("\n" + "=" * 60)
    print("PHASE 0: Data Loading & Feature Extraction")
    print("=" * 60)

    print("\n1a. Loading training data from hard_negatives.json...")
    af_samples, hn_samples = load_hard_negatives()
    n_af = len(af_samples)
    n_hn = len(hn_samples)
    print(f"    Selected {n_af} AF + {n_hn} HN = {n_af + n_hn} total")

    print("\n1b. Loading gold_106.json for Phase 4 eval...")
    gold_af, gold_aligned = load_gold_106()

    # ---- 2. Load Model + SAE ----
    print("\n2. Loading model and SAE...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   GPU VRAM: {vram_gb:.1f} GB")
    else:
        vram_gb = 0

    if vram_gb >= 40:
        print("   Loading full bfloat16 on GPU...")
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        print("   Loading bfloat16 on CPU (~54GB RAM)...")
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    model.eval()

    if hasattr(model, "device"):
        device = str(model.device)
    print(f"   Model loaded on: {device}")

    sae = ResidPostSAE(CONFIG["sae_repo"], CONFIG["sae_layer"])
    sae.to(device, dtype=torch.bfloat16)

    # ---- 3. Extract SAE Features ----
    print("\n3a. Extracting SAE features for training data...")
    all_train_texts = [s["text"] for s in af_samples] + [s["text"] for s in hn_samples]
    all_train_features = extract_sae_features(
        model, tokenizer, sae, all_train_texts, CONFIG["sae_layer"], device
    )
    X_af = all_train_features[:n_af]
    X_hn = all_train_features[n_af:]
    print(f"    AF features: {X_af.shape}, HN features: {X_hn.shape}")

    print("\n3b. Extracting SAE features for gold_106 eval...")
    gold_texts = [s["text"] for s in gold_af] + [s["text"] for s in gold_aligned]
    gold_features = extract_sae_features(
        model, tokenizer, sae, gold_texts, CONFIG["sae_layer"], device
    )
    n_gold_af = len(gold_af)
    X_gold_af = gold_features[:n_gold_af]
    X_gold_aligned = gold_features[n_gold_af:]
    print(f"    gold AF: {X_gold_af.shape}, gold aligned: {X_gold_aligned.shape}")

    # Free model memory
    del model, sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("   Model unloaded, memory freed.")

    # Build combined arrays
    X_all = np.vstack([X_af, X_hn])
    y_all = np.array([1] * n_af + [0] * n_hn)
    n_features = X_all.shape[1]
    print(f"\n   Combined: {X_all.shape}, {n_features} SAE features")

    # Train/test split (60/40)
    np.random.seed(CONFIG["seed"])
    af_perm = np.random.permutation(n_af)
    hn_perm = np.random.permutation(n_hn)
    n_train_per_class = int(0.6 * min(n_af, n_hn))
    n_test_per_class = min(n_af, n_hn) - n_train_per_class

    train_af_idx = af_perm[:n_train_per_class]
    test_af_idx = af_perm[n_train_per_class:n_train_per_class + n_test_per_class]
    train_hn_idx = hn_perm[:n_train_per_class]
    test_hn_idx = hn_perm[n_train_per_class:n_train_per_class + n_test_per_class]

    X_train = np.vstack([X_af[train_af_idx], X_hn[train_hn_idx]])
    y_train = np.array([1] * len(train_af_idx) + [0] * len(train_hn_idx))
    X_test = np.vstack([X_af[test_af_idx], X_hn[test_hn_idx]])
    y_test = np.array([1] * len(test_af_idx) + [0] * len(test_hn_idx))

    print(f"   Train: {len(train_af_idx)} AF + {len(train_hn_idx)} HN = {len(y_train)}")
    print(f"   Test:  {len(test_af_idx)} AF + {len(test_hn_idx)} HN = {len(y_test)}")

    # ============================================================
    # PHASE 1: Gradient Attribution
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Gradient Attribution")
    print("=" * 60)

    # Train full probe (all 16K features) as baseline
    print("\n4a. Training baseline probe (all features)...")
    baseline_probe = LogisticRegression(
        random_state=42, max_iter=1000, class_weight="balanced", solver="lbfgs"
    )
    baseline_probe.fit(np.nan_to_num(X_train), y_train)
    baseline_result = train_restricted_probe(
        X_train, y_train, X_test, y_test,
        np.arange(n_features), label="baseline_all_features"
    )
    print(f"    Baseline: FPR={baseline_result['fpr']:.2%}, "
          f"TPR={baseline_result['tpr']:.2%}, "
          f"AUROC={baseline_result['auroc']:.3f}")

    # Gradient attribution
    print("\n4b. Computing gradient attribution...")
    gradient_scores = compute_gradient_attribution(X_train, y_train, baseline_probe)
    print(f"    Non-zero gradient features: {(gradient_scores > 0).sum()}")
    print(f"    Max gradient: {gradient_scores.max():.6f}")
    print(f"    Mean gradient (non-zero): {gradient_scores[gradient_scores > 0].mean():.6f}")

    # Magnitude ranking for comparison
    print("\n4c. Computing magnitude ranking (Exp 02 approach)...")
    magnitude_scores = compute_magnitude_ranking(X_af, X_hn)

    # Compare
    print("\n4d. Comparing gradient vs magnitude rankings...")
    ranking_comparison = compare_rankings(gradient_scores, magnitude_scores, top_k=200)
    print(f"    Spearman rho: {ranking_comparison['spearman_rho']:.3f} "
          f"(p={ranking_comparison['spearman_pval']:.2e})")
    print(f"    Top-200 overlap: {ranking_comparison['top_200_overlap']}/200 "
          f"(Jaccard={ranking_comparison['top_200_jaccard']:.3f})")
    print(f"    Gradient top-10 IDs: {ranking_comparison['gradient_top_10']}")
    print(f"    Magnitude top-10 IDs: {ranking_comparison['magnitude_top_10']}")

    if ranking_comparison["spearman_rho"] < 0.5:
        print("    >> LOW correlation: gradient finds DIFFERENT features than magnitude!")
    else:
        print("    >> HIGH correlation: gradient and magnitude agree on feature importance.")

    # ============================================================
    # PHASE 2: Bootstrap Stability
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Bootstrap Stability Analysis")
    print("=" * 60)

    print(f"\n5. Running {CONFIG['n_bootstrap']} bootstrap iterations...")
    bootstrap_results = run_bootstrap(
        X_train, y_train, CONFIG["n_bootstrap"], CONFIG["top_k_values"]
    )

    # Report stability results
    for k in CONFIG["top_k_values"]:
        info = bootstrap_results["stability"][f"top_{k}"]
        print(f"\n    Top-{k}: {info['n_stable']} features stable "
              f"(appear in {CONFIG['stability_threshold']:.0%}+ of bootstraps)")
        print(f"    Mean stability of active features: {info['mean_stability']:.3f}")

    # Pick the top-K that gives a reasonable robust set
    # Use top_100 as default (middle ground)
    default_k = 100
    stability_scores = bootstrap_results["stability"][f"top_{default_k}"]["stability_scores"]
    stable_feature_ids = np.array(
        bootstrap_results["stability"][f"top_{default_k}"]["stable_feature_ids"]
    )
    n_stable = len(stable_feature_ids)
    print(f"\n    Using top-{default_k} stability: {n_stable} robust features "
          f"(vs Exp 02's 2,382 magnitude features)")

    if n_stable > 0 and n_stable < 2382:
        reduction = (1 - n_stable / 2382) * 100
        print(f"    Feature reduction: {reduction:.0f}% fewer features")

    # ============================================================
    # PHASE 3: Proxy Distillation & Ablation
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Proxy Distillation & Ablation Cascade")
    print("=" * 60)

    print("\n6a. Running ablation cascade on test set...")
    ablation_results = run_ablation_cascade(
        X_train, y_train, X_test, y_test,
        bootstrap_results["gradient_mean"],
        stability_scores,
        CONFIG["ablation_sizes"],
    )

    print("\n    | Strategy | N Features | FPR | TPR | AUROC |")
    print("    |----------|-----------|-----|-----|-------|")
    for r in ablation_results:
        print(f"    | {r['label']:30s} | {r['n_features']:9d} | "
              f"{r['fpr']:.0%} | {r['tpr']:.0%} | {r['auroc']:.3f} |")

    # 5-fold CV on best configurations
    print("\n6b. Running 5-fold CV on key configurations...")
    cv_results = []

    # All features baseline
    cv_all = run_cross_validation(
        X_all, y_all, np.arange(n_features), CONFIG["n_folds"],
        label="all_16384_features"
    )
    cv_results.append(cv_all)
    print(f"    All features CV: FPR={cv_all['fpr_mean']:.0%} +/- {cv_all['fpr_std']:.0%}")

    # Gradient top-50
    grad_ranked = np.argsort(-bootstrap_results["gradient_mean"])
    cv_grad50 = run_cross_validation(
        X_all, y_all, grad_ranked[:50], CONFIG["n_folds"],
        label="gradient_top_50"
    )
    cv_results.append(cv_grad50)
    print(f"    Gradient top-50 CV: FPR={cv_grad50['fpr_mean']:.0%} +/- {cv_grad50['fpr_std']:.0%}")

    # Stable features
    if n_stable > 0:
        cv_stable = run_cross_validation(
            X_all, y_all, stable_feature_ids, CONFIG["n_folds"],
            label=f"stable_{n_stable}_features"
        )
        cv_results.append(cv_stable)
        print(f"    Stable features CV: FPR={cv_stable['fpr_mean']:.0%} "
              f"+/- {cv_stable['fpr_std']:.0%}")

    # ============================================================
    # PHASE 4: gold_106 Evaluation (EVAL ONLY)
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 4: gold_106.json Evaluation (EVAL ONLY)")
    print("=" * 60)

    X_gold = np.vstack([X_gold_af, X_gold_aligned])
    y_gold = np.array([1] * len(X_gold_af) + [0] * len(X_gold_aligned))

    print(f"\n7. Evaluating on gold_106: {len(X_gold_af)} AF + {len(X_gold_aligned)} aligned")

    gold_eval_results = []

    # All features
    r = train_restricted_probe(
        X_train, y_train, X_gold, y_gold,
        np.arange(n_features), label="gold_all_features"
    )
    gold_eval_results.append(r)
    print(f"    All features: FPR={r['fpr']:.0%}, TPR={r['tpr']:.0%}, AUROC={r['auroc']:.3f}")

    # Gradient top-50
    r = train_restricted_probe(
        X_train, y_train, X_gold, y_gold,
        grad_ranked[:50], label="gold_gradient_top_50"
    )
    gold_eval_results.append(r)
    print(f"    Gradient top-50: FPR={r['fpr']:.0%}, TPR={r['tpr']:.0%}, AUROC={r['auroc']:.3f}")

    # Stable features
    if n_stable > 0:
        r = train_restricted_probe(
            X_train, y_train, X_gold, y_gold,
            stable_feature_ids, label=f"gold_stable_{n_stable}"
        )
        gold_eval_results.append(r)
        print(f"    Stable ({n_stable}): FPR={r['fpr']:.0%}, TPR={r['tpr']:.0%}, "
              f"AUROC={r['auroc']:.3f}")

    # Ablation on gold_106
    for k in [5, 10, 20]:
        r = train_restricted_probe(
            X_train, y_train, X_gold, y_gold,
            grad_ranked[:k], label=f"gold_gradient_top_{k}"
        )
        gold_eval_results.append(r)
        print(f"    Gradient top-{k}: FPR={r['fpr']:.0%}, TPR={r['tpr']:.0%}, "
              f"AUROC={r['auroc']:.3f}")

    # ============================================================
    # Feature Inspection
    # ============================================================
    print("\n" + "=" * 60)
    print("FEATURE INSPECTION: Top Gradient-Attributed Features")
    print("=" * 60)

    top_grad_ids = grad_ranked[:20]
    feature_reports = inspect_top_features(
        X_af, X_hn, top_grad_ids,
        bootstrap_results["gradient_mean"], stability_scores
    )

    print(f"\n    | Rank | Feature ID | Gradient | Stability | Cohen's d | Direction |")
    print(f"    |------|-----------|----------|-----------|-----------|-----------|")
    for i, fr in enumerate(feature_reports):
        print(f"    | {i+1:4d} | {fr['feature_id']:9d} | {fr['gradient_score']:.6f} | "
              f"{fr['stability_score']:.3f}     | {fr['cohens_d']:+.3f}    | "
              f"{fr['direction']:9s} |")

    # ============================================================
    # COMPARISON TABLE
    # ============================================================
    print("\n\n" + "=" * 80)
    print("FINAL COMPARISON TABLE")
    print("=" * 80)

    print("\n| Method | FPR | TPR | Features | Source |")
    print("|--------|-----|-----|----------|--------|")
    print("| SAE cluster probe (Exp 01) | 100% | ~88% | 500 clusters | Previous |")
    print("| SAE magnitude (Exp 02-04) | 91% | ~95% | 2,382 | Previous |")
    print("| DAG patterns (Exp 07) | 28% | 92% | N/A (structural) | Previous |")
    print(f"| All SAE features (this exp) | {baseline_result['fpr']:.0%} | "
          f"{baseline_result['tpr']:.0%} | 16,384 | Exp 09 |")

    if len(ablation_results) > 0:
        # Find best ablation result
        best_abl = min(ablation_results, key=lambda r: r["fpr"])
        print(f"| {best_abl['label']} | {best_abl['fpr']:.0%} | "
              f"{best_abl['tpr']:.0%} | {best_abl['n_features']} | Exp 09 |")

    if n_stable > 0:
        stable_abl = [r for r in ablation_results if "stable" in r["label"]]
        if stable_abl:
            best_stable = min(stable_abl, key=lambda r: r["fpr"])
            print(f"| {best_stable['label']} | {best_stable['fpr']:.0%} | "
                  f"{best_stable['tpr']:.0%} | {best_stable['n_features']} | Exp 09 |")

    # ============================================================
    # INTERPRETATION
    # ============================================================
    print("\n\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    best_fpr = baseline_result["fpr"]
    if ablation_results:
        best_abl = min(ablation_results, key=lambda r: r["fpr"])
        if best_abl["fpr"] < best_fpr:
            best_fpr = best_abl["fpr"]
            best_method = best_abl["label"]
        else:
            best_method = "all_features"
    else:
        best_method = "all_features"

    rho = ranking_comparison["spearman_rho"]

    if best_fpr < 0.50 and rho < 0.5:
        print(f"\nSTRONG SUCCESS: Gradient attribution achieves {best_fpr:.0%} FPR")
        print(f"  Method: {best_method}")
        print(f"  Gradient vs magnitude correlation: rho={rho:.3f} (different features!)")
        print(f"  Stable features: {n_stable}")
        print("  -> The intent signal WAS in the SAE all along.")
        print("  -> Activation magnitude selected RLHF-awareness features (shared).")
        print("  -> Gradient attribution selects decision-relevant features (AF-specific).")
    elif best_fpr < 0.50:
        print(f"\nMODERATE SUCCESS: FPR improved to {best_fpr:.0%}")
        print(f"  Method: {best_method}")
        print(f"  But gradient/magnitude rankings are correlated (rho={rho:.3f}).")
        print("  -> Feature SELECTION method matters, but it's finding similar features.")
        print("  -> The improvement may come from dimensionality reduction (regularization).")
    elif best_fpr < 0.70:
        print(f"\nPARTIAL: Some improvement ({best_fpr:.0%} FPR) but still high")
        print(f"  Method: {best_method}")
        print("  -> Gradient attribution helps but doesn't overcome the fundamental issue.")
        print("  -> SAE captures some intent signal but shared RLHF-awareness dominates.")
    else:
        print(f"\nFAILURE: Best FPR is {best_fpr:.0%} — no improvement over Exp 04")
        print("  -> The problem IS the SAE representation, not feature selection.")
        print("  -> Confirms Exp 06-07: intent lives in reasoning structure (edges), not features.")
        print("  -> Gradient attribution cannot extract signal the SAE doesn't encode.")

    print(f"\n  Bootstrap stability summary:")
    print(f"    Robust features (stability >= {CONFIG['stability_threshold']:.0%}): {n_stable}")
    print(f"    Exp 02 magnitude features: 2,382")
    if n_stable > 0:
        print(f"    Reduction: {(1 - n_stable / 2382) * 100:.0f}%")

    # ============================================================
    # Save Results
    # ============================================================
    results = {
        "config": {
            "model": CONFIG["model_name"],
            "sae_layer": CONFIG["sae_layer"],
            "n_af": n_af,
            "n_hn": n_hn,
            "n_bootstrap": CONFIG["n_bootstrap"],
            "stability_threshold": CONFIG["stability_threshold"],
            "seed": CONFIG["seed"],
        },
        "phase1_gradient_attribution": {
            "baseline": baseline_result,
            "ranking_comparison": ranking_comparison,
            "gradient_top_20_ids": grad_ranked[:20].tolist(),
            "n_nonzero_gradient": int((gradient_scores > 0).sum()),
        },
        "phase2_bootstrap": {
            "n_bootstrap": CONFIG["n_bootstrap"],
            "stability_threshold": CONFIG["stability_threshold"],
            "results_per_k": {
                f"top_{k}": {
                    "n_stable": bootstrap_results["stability"][f"top_{k}"]["n_stable"],
                    "stable_feature_ids": bootstrap_results["stability"][f"top_{k}"]["stable_feature_ids"],
                    "mean_stability": bootstrap_results["stability"][f"top_{k}"]["mean_stability"],
                }
                for k in CONFIG["top_k_values"]
            },
        },
        "phase3_ablation": ablation_results,
        "phase3_cv": [
            {k: v for k, v in cv.items() if k != "folds"}
            for cv in cv_results
        ],
        "phase4_gold_106_eval": gold_eval_results,
        "feature_inspection": feature_reports,
        "comparison": {
            "sae_cluster_probe_exp01": {"fpr": 1.0, "source": "Exp 01"},
            "sae_magnitude_exp04": {"fpr": 0.91, "source": "Exp 04"},
            "dag_patterns_exp07": {"fpr": 0.28, "source": "Exp 07"},
            "best_this_exp": {
                "fpr": best_fpr,
                "method": best_method,
                "n_stable_features": n_stable,
                "source": "Exp 09",
            },
        },
    }

    # Save robust features separately for downstream use
    robust_features = {
        "description": "Bootstrap-stable SAE features from Experiment 09",
        "stability_threshold": CONFIG["stability_threshold"],
        "n_bootstrap": CONFIG["n_bootstrap"],
        "top_k_used": default_k,
        "features": [],
    }
    if n_stable > 0:
        for fid in stable_feature_ids:
            robust_features["features"].append({
                "feature_id": int(fid),
                "stability_score": float(stability_scores[fid]),
                "gradient_score": float(bootstrap_results["gradient_mean"][fid]),
            })
        # Sort by gradient score descending
        robust_features["features"].sort(key=lambda x: -x["gradient_score"])

    output_dir = CONFIG["output_dir"]
    results_path = output_dir / "results.json"
    robust_path = output_dir / "robust_features.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {results_path}")

    with open(robust_path, "w") as f:
        json.dump(robust_features, f, indent=2)
    print(f"Robust features saved to: {robust_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
