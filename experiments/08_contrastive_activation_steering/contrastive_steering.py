#!/usr/bin/env python3
"""
Experiment 08: CASAL-Adapted Contrastive Activation Steering for AF Detection

Adapts the CASAL (Contrastive Activation Steering for Amortized Learning) approach
to alignment faking detection. Instead of SAE features (16K sparse), uses raw
residual stream activations with contrastive steering vectors.

CASAL adaptation:
- Step 1 (Probe): AF vs HN labels already exist (no sampling needed)
- Step 2 (Steer): Compute contrastive vector v = mean(AF) - mean(HN) at multiple layers
- Classification: Cosine similarity to v + linear probe on raw activations

Key CASAL insight: Extract at LAST TOKEN position (not max-pool).
We test both pooling strategies.

Requires GPU (~50GB VRAM for Gemma-3-27B in bfloat16).
"""

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    silhouette_score,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

CONFIG = {
    "model_name": "google/gemma-3-27b-it",
    "sae_repo": "google/gemma-scope-2-27b-it",
    "sae_layer": 40,
    "layers_to_probe": [10, 15, 20, 25, 30, 35, 40],
    "max_length": 2048,
    "n_af": 25,
    "n_hn": 25,
    "n_train": 15,  # per class
    "n_test": 10,   # per class
    "seed": 42,
    "n_folds": 5,
    # AF samples come from hard_negatives.json (has 1,239 AF samples)
    # gold_106.json is EVAL-ONLY and must NOT be used for training
    "hard_neg_path": "~/alignment-faking-training/hard_negatives.json",
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

def load_data() -> Tuple[List[Dict], List[Dict]]:
    """Load AF and HN samples.

    IMPORTANT: gold_106.json is eval-only (hand-labeled). We use AF samples
    from hard_negatives.json (which contains 1,239 AF samples) for training
    the contrastive vector, and reserve gold_106 AF for evaluation only.
    """
    # Load hard_negatives.json (contains AF, aligned, AND hard_negative samples)
    hn_path = Path(CONFIG["hard_neg_path"]).expanduser()
    with open(hn_path) as f:
        hn_data = json.load(f)

    all_samples = hn_data["samples"] if "samples" in hn_data else hn_data
    if isinstance(all_samples, dict):
        all_samples = list(all_samples.values())

    # AF samples from hard_negatives.json (NOT gold_106 — that's eval-only)
    af_all = [s for s in all_samples if s.get("label") == "alignment_faking"]
    print(f"  AF source (hard_negatives.json): {len(af_all)} alignment_faking samples")

    # HN samples
    hn_all = [s for s in all_samples if s.get("label") == "hard_negative"]
    print(f"  HN source (hard_negatives.json): {len(hn_all)} hard_negative samples")

    # Sample
    np.random.seed(CONFIG["seed"])
    af_indices = np.random.choice(len(af_all), CONFIG["n_af"], replace=False)
    hn_indices = np.random.choice(len(hn_all), CONFIG["n_hn"], replace=False)

    af_selected = [af_all[i] for i in af_indices]
    hn_selected = [hn_all[i] for i in hn_indices]

    return af_selected, hn_selected


# ============================================================
# Activation Extraction
# ============================================================

@torch.no_grad()
def extract_all_activations(
    model, tokenizer, texts: List[str], layers: List[int], device: str
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Extract activations at multiple layers for all texts.
    Returns dict with 'last_token' and 'max_pool' keys,
    each mapping layer_idx -> np.array([n_samples, hidden_dim]).
    """
    n = len(texts)
    # Gemma-3 uses nested config
    if hasattr(model.config, "text_config"):
        hidden_dim = model.config.text_config.hidden_size
    else:
        hidden_dim = model.config.hidden_size

    # Pre-allocate
    activations = {
        "last_token": {l: np.zeros((n, hidden_dim), dtype=np.float32) for l in layers},
        "max_pool": {l: np.zeros((n, hidden_dim), dtype=np.float32) for l in layers},
    }

    for i, text in enumerate(tqdm(texts, desc="Extracting activations")):
        inputs = tokenizer(
            text, return_tensors="pt", max_length=CONFIG["max_length"], truncation=True
        )
        # With device_map="auto" or split maps, use model's device for inputs
        if hasattr(model, "hf_device_map"):
            inputs = inputs.to(model.device)
        else:
            inputs = inputs.to(device)

        outputs = model(**inputs, output_hidden_states=True)

        for l in layers:
            # hidden_states[0] = input embeddings, [1] = layer 0 output, ...
            hidden = outputs.hidden_states[l + 1]  # [1, seq_len, hidden_dim]

            # Last token (CASAL approach)
            last_tok = hidden[:, -1, :].cpu().float().numpy()[0]
            # Replace NaN with 0 (can occur with quantized/offloaded layers)
            np.nan_to_num(last_tok, copy=False)
            activations["last_token"][l][i] = last_tok

            # Max pool (existing baseline)
            # Replace NaN before max to avoid propagation
            hidden_clean = torch.nan_to_num(hidden, nan=0.0)
            max_pooled = hidden_clean.max(dim=1)[0].cpu().float().numpy()[0]
            activations["max_pool"][l][i] = max_pooled

        # Free GPU memory
        del outputs
        if i % 10 == 0:
            torch.cuda.empty_cache()

    return activations


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

        # SAE encode — move hidden to SAE device
        sae_acts = sae.encode(hidden[0].to(sae.w_enc.device).to(sae.w_enc.dtype))  # [seq_len, n_features]
        max_acts = sae_acts.max(dim=0)[0].cpu().float().numpy()  # [n_features]
        features[i] = max_acts

        del outputs
        if i % 10 == 0:
            torch.cuda.empty_cache()

    return features


# ============================================================
# Contrastive Vector
# ============================================================

def compute_contrastive_vector(af_acts: np.ndarray, hn_acts: np.ndarray) -> np.ndarray:
    """Compute normalized contrastive vector: mean(AF) - mean(HN)."""
    v = af_acts.mean(axis=0) - hn_acts.mean(axis=0)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v


# ============================================================
# Evaluation Functions
# ============================================================

def evaluate_contrastive(
    v: np.ndarray, test_af: np.ndarray, test_hn: np.ndarray
) -> Dict:
    """Evaluate contrastive cosine similarity classification."""
    # Normalize test samples
    af_norms = np.linalg.norm(test_af, axis=1, keepdims=True)
    hn_norms = np.linalg.norm(test_hn, axis=1, keepdims=True)
    af_normed = test_af / np.maximum(af_norms, 1e-8)
    hn_normed = test_hn / np.maximum(hn_norms, 1e-8)

    # Cosine similarity with contrastive vector
    af_cos = af_normed @ v  # should be positive
    hn_cos = hn_normed @ v  # should be negative

    # Classification: positive cosine = AF
    af_preds = (af_cos > 0).astype(int)  # TP if 1
    hn_preds = (hn_cos > 0).astype(int)  # FP if 1

    tpr = af_preds.mean()
    fpr = hn_preds.mean()
    n_af, n_hn = len(test_af), len(test_hn)
    accuracy = (af_preds.sum() + (n_hn - hn_preds.sum())) / (n_af + n_hn)

    # AUROC using cosine scores
    all_scores = np.concatenate([af_cos, hn_cos])
    all_labels = np.array([1] * n_af + [0] * n_hn)
    try:
        auroc = float(roc_auc_score(all_labels, all_scores))
    except ValueError:
        auroc = 0.5

    return {
        "accuracy": float(accuracy),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "auroc": auroc,
        "af_cos_mean": float(af_cos.mean()),
        "hn_cos_mean": float(hn_cos.mean()),
    }


def evaluate_projection(
    v: np.ndarray, test_af: np.ndarray, test_hn: np.ndarray
) -> Dict:
    """Analyze projection of samples onto contrastive vector."""
    af_proj = test_af @ v
    hn_proj = test_hn @ v

    af_mean = float(af_proj.mean())
    hn_mean = float(hn_proj.mean())
    af_std = float(af_proj.std())
    hn_std = float(hn_proj.std())

    # Separation: difference of means / pooled std
    pooled_std = np.sqrt((af_std**2 + hn_std**2) / 2)
    separation = (af_mean - hn_mean) / pooled_std if pooled_std > 0 else 0.0

    # Overlap: fraction of HN projecting above AF mean
    overlap_hn_above = float((hn_proj > af_mean).mean())
    overlap_af_below = float((af_proj < hn_mean).mean())

    return {
        "af_proj_mean": af_mean,
        "hn_proj_mean": hn_mean,
        "af_proj_std": af_std,
        "hn_proj_std": hn_std,
        "separation_d": float(separation),
        "overlap_hn_above_af_mean": overlap_hn_above,
        "overlap_af_below_hn_mean": overlap_af_below,
    }


def evaluate_linear_probe(
    train_X: np.ndarray, train_y: np.ndarray,
    test_X: np.ndarray, test_y: np.ndarray
) -> Dict:
    """Train logistic regression and evaluate."""
    # Clean NaN values
    train_X = np.nan_to_num(train_X)
    test_X = np.nan_to_num(test_X)

    probe = LogisticRegression(
        random_state=42, max_iter=1000, class_weight="balanced", solver="lbfgs"
    )
    probe.fit(train_X, train_y)

    preds = probe.predict(test_X)
    probs = probe.predict_proba(test_X)[:, 1]

    # Compute metrics
    n_af = int(test_y.sum())
    n_hn = len(test_y) - n_af

    tp = int(((preds == 1) & (test_y == 1)).sum())
    fp = int(((preds == 1) & (test_y == 0)).sum())

    tpr = tp / n_af if n_af > 0 else 0.0
    fpr = fp / n_hn if n_hn > 0 else 0.0
    accuracy = float(accuracy_score(test_y, preds))

    try:
        auroc = float(roc_auc_score(test_y, probs))
    except ValueError:
        auroc = 0.5

    return {
        "accuracy": accuracy,
        "fpr": fpr,
        "tpr": tpr,
        "auroc": auroc,
    }


def evaluate_silhouette(test_X: np.ndarray, test_y: np.ndarray) -> float:
    """Compute silhouette score for cluster separation."""
    if len(np.unique(test_y)) < 2:
        return 0.0
    try:
        return float(silhouette_score(test_X, test_y))
    except ValueError:
        return 0.0


# ============================================================
# Cross-Validation
# ============================================================

def run_cross_validation(
    af_acts: np.ndarray, hn_acts: np.ndarray, n_folds: int = 5
) -> Dict:
    """Run n-fold CV. Recompute contrastive vector each fold."""
    n_af = len(af_acts)
    n_hn = len(hn_acts)
    fold_size_af = n_af // n_folds
    fold_size_hn = n_hn // n_folds

    fold_results = []

    for fold in range(n_folds):
        # Split
        test_af_idx = list(range(fold * fold_size_af, (fold + 1) * fold_size_af))
        test_hn_idx = list(range(fold * fold_size_hn, (fold + 1) * fold_size_hn))
        train_af_idx = [i for i in range(n_af) if i not in test_af_idx]
        train_hn_idx = [i for i in range(n_hn) if i not in test_hn_idx]

        train_af = af_acts[train_af_idx]
        train_hn = hn_acts[train_hn_idx]
        test_af = af_acts[test_af_idx]
        test_hn = hn_acts[test_hn_idx]

        # Contrastive vector from train
        v = compute_contrastive_vector(train_af, train_hn)
        contrastive = evaluate_contrastive(v, test_af, test_hn)

        # Linear probe from train
        train_X = np.vstack([train_af, train_hn])
        train_y = np.array([1] * len(train_af) + [0] * len(train_hn))
        test_X = np.vstack([test_af, test_hn])
        test_y = np.array([1] * len(test_af) + [0] * len(test_hn))
        probe = evaluate_linear_probe(train_X, train_y, test_X, test_y)

        fold_results.append({
            "fold": fold + 1,
            "contrastive": contrastive,
            "linear_probe": probe,
        })

    # Aggregate
    def mean_std(key_path):
        vals = []
        for fr in fold_results:
            obj = fr
            for k in key_path.split("."):
                obj = obj[k]
            vals.append(obj)
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    summary = {
        "folds": fold_results,
        "contrastive_accuracy": mean_std("contrastive.accuracy"),
        "contrastive_fpr": mean_std("contrastive.fpr"),
        "contrastive_tpr": mean_std("contrastive.tpr"),
        "contrastive_auroc": mean_std("contrastive.auroc"),
        "probe_accuracy": mean_std("linear_probe.accuracy"),
        "probe_fpr": mean_std("linear_probe.fpr"),
        "probe_tpr": mean_std("linear_probe.tpr"),
        "probe_auroc": mean_std("linear_probe.auroc"),
    }

    return summary


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("Experiment 08: CASAL-Adapted Contrastive Activation Steering")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cpu":
        print("WARNING: Running on CPU. This will be very slow for 27B model.")
        print("Consider running on GPU (nigel.birs.ca).")

    # ---- 1. Load Data ----
    print("\n1. Loading data...")
    af_samples, hn_samples = load_data()
    all_texts = [s["text"] for s in af_samples] + [s["text"] for s in hn_samples]
    n_af = len(af_samples)
    n_hn = len(hn_samples)
    print(f"   Selected {n_af} AF + {n_hn} HN = {n_af + n_hn} total")

    # Train/test split indices
    np.random.seed(CONFIG["seed"])
    af_perm = np.random.permutation(n_af)
    hn_perm = np.random.permutation(n_hn)

    train_af_idx = af_perm[: CONFIG["n_train"]]
    test_af_idx = af_perm[CONFIG["n_train"] :]
    train_hn_idx = hn_perm[: CONFIG["n_train"]]
    test_hn_idx = hn_perm[CONFIG["n_train"] :]

    print(f"   Split: {CONFIG['n_train']} train + {CONFIG['n_test']} test per class")

    # ---- 2. Load Model ----
    print("\n2. Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    # Check available VRAM to decide loading strategy
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   GPU VRAM: {vram_gb:.1f} GB")
    else:
        vram_gb = 0

    if vram_gb >= 40:
        # Full bfloat16 on GPU for >=40GB VRAM (A100, etc.)
        print("   Loading full bfloat16 on GPU...")
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        # CPU-only bfloat16: 27B model = ~54GB RAM, numerically stable
        # Note: 4-bit quantization with CPU offload produces NaN in hidden states
        # due to cross-device transfer issues. CPU-only is slower but correct.
        print("   GPU too small for 27B model. Loading bfloat16 on CPU (~54GB RAM)...")
        print("   (4-bit quantization with CPU offload produces NaN - using pure CPU)")
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

    model.eval()
    print(f"   Model loaded: {CONFIG['model_name']}")
    # Gemma-3 uses nested config: text_config.hidden_size
    if hasattr(model.config, "text_config"):
        hidden_dim = model.config.text_config.hidden_size
    else:
        hidden_dim = model.config.hidden_size

    # Update device to match actual model placement
    if hasattr(model, "device"):
        device = str(model.device)
    print(f"   Hidden dim: {hidden_dim}")
    print(f"   Actual device: {device}")

    # ---- 3. Extract Activations ----
    print("\n3. Extracting activations at layers:", CONFIG["layers_to_probe"])
    activations = extract_all_activations(
        model, tokenizer, all_texts, CONFIG["layers_to_probe"], device
    )
    print("   Done. Shapes and NaN check:")
    for pool_mode in ["last_token", "max_pool"]:
        for l in CONFIG["layers_to_probe"]:
            arr = activations[pool_mode][l]
            nan_count = np.isnan(arr).sum()
            nan_pct = nan_count / arr.size * 100
            print(f"     {pool_mode} layer {l}: {arr.shape} | NaN: {nan_count} ({nan_pct:.1f}%)")

    # ---- 4. SAE Baseline ----
    print("\n4. Loading SAE for baseline comparison...")
    sae = ResidPostSAE(CONFIG["sae_repo"], CONFIG["sae_layer"])
    sae.to(device, dtype=torch.bfloat16)
    sae_features = extract_sae_features(
        model, tokenizer, sae, all_texts, CONFIG["sae_layer"], device
    )
    print(f"   SAE features shape: {sae_features.shape}")

    # Free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc; gc.collect()
    print("   Model unloaded, memory freed.")

    # ---- 5. Evaluate Each Layer ----
    print("\n5. Evaluating per layer...")

    results = {
        "config": {
            "model": CONFIG["model_name"],
            "layers": CONFIG["layers_to_probe"],
            "n_af": n_af,
            "n_hn": n_hn,
            "n_train": CONFIG["n_train"],
            "n_test": CONFIG["n_test"],
            "seed": CONFIG["seed"],
            "pooling_modes": ["last_token", "max_pool"],
        },
        "single_split": {},
        "cross_validation": {},
        "sae_baseline": {},
    }

    for pool_mode in ["last_token", "max_pool"]:
        print(f"\n  === Pooling: {pool_mode} ===")
        results["single_split"][pool_mode] = {}
        results["cross_validation"][pool_mode] = {}

        for layer in CONFIG["layers_to_probe"]:
            print(f"\n  Layer {layer}:")
            acts = activations[pool_mode][layer]

            # Split into AF / HN portions (first n_af are AF, rest are HN)
            af_acts = acts[:n_af]
            hn_acts = acts[n_af:]

            # Train/test
            train_af = af_acts[train_af_idx]
            test_af = af_acts[test_af_idx]
            train_hn = hn_acts[train_hn_idx]
            test_hn = hn_acts[test_hn_idx]

            # Contrastive vector from train
            v = compute_contrastive_vector(train_af, train_hn)

            # Evaluate contrastive
            contrastive = evaluate_contrastive(v, test_af, test_hn)
            print(f"    Contrastive: acc={contrastive['accuracy']:.2f} fpr={contrastive['fpr']:.2f} tpr={contrastive['tpr']:.2f} auroc={contrastive['auroc']:.3f}")

            # Projection analysis
            projection = evaluate_projection(v, test_af, test_hn)
            print(f"    Projection: separation_d={projection['separation_d']:.2f}")

            # Linear probe
            train_X = np.vstack([train_af, train_hn])
            train_y = np.array([1] * len(train_af) + [0] * len(train_hn))
            test_X = np.vstack([test_af, test_hn])
            test_y = np.array([1] * len(test_af) + [0] * len(test_hn))

            probe = evaluate_linear_probe(train_X, train_y, test_X, test_y)
            print(f"    Probe: acc={probe['accuracy']:.2f} fpr={probe['fpr']:.2f} tpr={probe['tpr']:.2f} auroc={probe['auroc']:.3f}")

            # Silhouette
            sil = evaluate_silhouette(test_X, test_y)
            print(f"    Silhouette: {sil:.3f}")

            results["single_split"][pool_mode][f"layer_{layer}"] = {
                "contrastive": contrastive,
                "projection": projection,
                "linear_probe": probe,
                "silhouette": sil,
            }

            # Cross-validation
            cv = run_cross_validation(af_acts, hn_acts, CONFIG["n_folds"])
            print(f"    CV contrastive FPR: {cv['contrastive_fpr']['mean']:.2f} +/- {cv['contrastive_fpr']['std']:.2f}")
            print(f"    CV probe FPR: {cv['probe_fpr']['mean']:.2f} +/- {cv['probe_fpr']['std']:.2f}")

            results["cross_validation"][pool_mode][f"layer_{layer}"] = cv

    # ---- 6. SAE Baseline ----
    print("\n\n  === SAE Baseline (layer 40, max-pool) ===")

    sae_af = sae_features[:n_af]
    sae_hn = sae_features[n_af:]

    sae_train_X = np.vstack([sae_af[train_af_idx], sae_hn[train_hn_idx]])
    sae_train_y = np.array([1] * len(train_af_idx) + [0] * len(train_hn_idx))
    sae_test_X = np.vstack([sae_af[test_af_idx], sae_hn[test_hn_idx]])
    sae_test_y = np.array([1] * len(test_af_idx) + [0] * len(test_hn_idx))

    sae_probe = evaluate_linear_probe(sae_train_X, sae_train_y, sae_test_X, sae_test_y)
    print(f"  SAE probe: acc={sae_probe['accuracy']:.2f} fpr={sae_probe['fpr']:.2f} tpr={sae_probe['tpr']:.2f} auroc={sae_probe['auroc']:.3f}")

    # SAE contrastive
    sae_v = compute_contrastive_vector(sae_af[train_af_idx], sae_hn[train_hn_idx])
    sae_contrastive = evaluate_contrastive(sae_v, sae_af[test_af_idx], sae_hn[test_hn_idx])
    print(f"  SAE contrastive: acc={sae_contrastive['accuracy']:.2f} fpr={sae_contrastive['fpr']:.2f}")

    # SAE CV
    sae_cv = run_cross_validation(sae_af, sae_hn, CONFIG["n_folds"])
    print(f"  SAE CV probe FPR: {sae_cv['probe_fpr']['mean']:.2f} +/- {sae_cv['probe_fpr']['std']:.2f}")

    results["sae_baseline"] = {
        "single_split": {
            "linear_probe": sae_probe,
            "contrastive": sae_contrastive,
        },
        "cross_validation": sae_cv,
    }

    # ---- 7. Comparison Table ----
    print("\n\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)

    # Find best layer for each pooling mode
    best_results = {}
    for pool_mode in ["last_token", "max_pool"]:
        best_layer = None
        best_fpr = 1.0
        for layer in CONFIG["layers_to_probe"]:
            cv = results["cross_validation"][pool_mode][f"layer_{layer}"]
            fpr = cv["probe_fpr"]["mean"]
            if fpr < best_fpr:
                best_fpr = fpr
                best_layer = layer
        best_results[pool_mode] = {
            "layer": best_layer,
            "fpr": best_fpr,
            "cv": results["cross_validation"][pool_mode][f"layer_{best_layer}"],
        }

    print("\n| Method | FPR | TPR | Accuracy | AUROC | Source |")
    print("|--------|-----|-----|----------|-------|--------|")
    print("| SAE cluster probe (Exp 01) | 100% | ~88% | ~50% | 0.884 | Previous |")
    print("| SAE minimal probe (Exp 04) | 91% | ~95% | ~52% | N/A | Previous |")
    print("| DAG patterns 5-fold (Exp 07) | 28% | 92% | 82% | N/A | Previous |")

    sae_cv_fpr = results["sae_baseline"]["cross_validation"]["probe_fpr"]["mean"]
    sae_cv_tpr = results["sae_baseline"]["cross_validation"]["probe_tpr"]["mean"]
    sae_cv_acc = results["sae_baseline"]["cross_validation"]["probe_accuracy"]["mean"]
    sae_cv_auroc = results["sae_baseline"]["cross_validation"]["probe_auroc"]["mean"]
    print(f"| SAE probe this run (layer 40) | {sae_cv_fpr:.0%} | {sae_cv_tpr:.0%} | {sae_cv_acc:.0%} | {sae_cv_auroc:.3f} | This exp |")

    for pool_mode in ["last_token", "max_pool"]:
        br = best_results[pool_mode]
        cv = br["cv"]
        label = "last-token (CASAL)" if pool_mode == "last_token" else "max-pool"
        print(
            f"| Raw probe {label} (L{br['layer']}) "
            f"| {cv['probe_fpr']['mean']:.0%} "
            f"| {cv['probe_tpr']['mean']:.0%} "
            f"| {cv['probe_accuracy']['mean']:.0%} "
            f"| {cv['probe_auroc']['mean']:.3f} "
            f"| This exp |"
        )

    for pool_mode in ["last_token", "max_pool"]:
        br = best_results[pool_mode]
        cv = br["cv"]
        label = "last-token (CASAL)" if pool_mode == "last_token" else "max-pool"
        print(
            f"| Contrastive {label} (L{br['layer']}) "
            f"| {cv['contrastive_fpr']['mean']:.0%} "
            f"| {cv['contrastive_tpr']['mean']:.0%} "
            f"| {cv['contrastive_accuracy']['mean']:.0%} "
            f"| {cv['contrastive_auroc']['mean']:.3f} "
            f"| This exp |"
        )

    # ---- 8. Layer-wise Detail ----
    print("\n\n" + "=" * 80)
    print("LAYER-WISE DETAIL (5-fold CV)")
    print("=" * 80)

    for pool_mode in ["last_token", "max_pool"]:
        print(f"\n  {pool_mode.upper()}:")
        print(f"  | Layer | Probe FPR | Probe TPR | Probe Acc | Contrastive FPR | Silhouette |")
        print(f"  |-------|-----------|-----------|-----------|-----------------|------------|")
        for layer in CONFIG["layers_to_probe"]:
            cv = results["cross_validation"][pool_mode][f"layer_{layer}"]
            ss = results["single_split"][pool_mode][f"layer_{layer}"]
            print(
                f"  | {layer:>5} "
                f"| {cv['probe_fpr']['mean']:>8.0%} "
                f"| {cv['probe_tpr']['mean']:>8.0%} "
                f"| {cv['probe_accuracy']['mean']:>8.0%} "
                f"| {cv['contrastive_fpr']['mean']:>14.0%} "
                f"| {ss['silhouette']:>9.3f} |"
            )

    # ---- 9. Interpretation ----
    print("\n\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    best_overall_fpr = min(best_results[p]["fpr"] for p in best_results)
    best_pool = min(best_results, key=lambda p: best_results[p]["fpr"])
    best_layer = best_results[best_pool]["layer"]

    if best_overall_fpr < 0.40:
        print(f"\nSTRONG SUCCESS: Raw activations achieve {best_overall_fpr:.0%} FPR")
        print(f"  Best: {best_pool} at layer {best_layer}")
        print("  -> SAE is LOSSY for intent detection.")
        print("  -> Intent IS linearly separable in raw residual stream.")
        if best_pool == "last_token":
            print("  -> CASAL's last-token extraction is effective for AF detection.")
    elif best_overall_fpr < 0.70:
        print(f"\nMODERATE SUCCESS: Raw activations achieve {best_overall_fpr:.0%} FPR")
        print(f"  Best: {best_pool} at layer {best_layer}")
        print("  -> Partial intent signal in raw activations.")
        print("  -> SAE loses some signal but raw activations alone aren't sufficient.")
    else:
        print(f"\nFAILURE: Raw activations achieve {best_overall_fpr:.0%} FPR")
        print("  -> Intent is NOT linearly separable in residual stream at any layer.")
        print("  -> Confirms 'edges not nodes' hypothesis (Exp 06-07).")
        print("  -> Intent lives in reasoning STRUCTURE, not activation SPACE.")

    # Layer emergence analysis
    print("\n  Layer emergence pattern:")
    for pool_mode in ["last_token", "max_pool"]:
        fprs = []
        for layer in CONFIG["layers_to_probe"]:
            cv = results["cross_validation"][pool_mode][f"layer_{layer}"]
            fprs.append(cv["probe_fpr"]["mean"])
        best_l = CONFIG["layers_to_probe"][np.argmin(fprs)]
        print(f"    {pool_mode}: best at layer {best_l} (FPR={min(fprs):.0%})")
        if best_l < 35:
            print(f"      -> Intent emerges BEFORE layer 40 (SAE layer). SAE misses the optimal layer.")
        elif best_l == 40:
            print(f"      -> Intent is at layer 40. SAE compression is the bottleneck.")

    # ---- 10. Comparison summary + Save ----
    results["comparison"] = {
        "sae_cluster_probe_exp01": {"fpr": 1.0, "tpr": 0.88, "source": "Exp 01"},
        "sae_minimal_probe_exp04": {"fpr": 0.91, "tpr": 0.95, "source": "Exp 04"},
        "dag_patterns_exp07": {"fpr": 0.28, "tpr": 0.92, "source": "Exp 07"},
        "sae_probe_this_exp": {
            "fpr": sae_cv_fpr,
            "tpr": sae_cv_tpr,
            "source": "Exp 08 SAE baseline",
        },
        "best_raw_probe": {
            "fpr": best_overall_fpr,
            "tpr": best_results[best_pool]["cv"]["probe_tpr"]["mean"],
            "pool_mode": best_pool,
            "layer": best_layer,
            "source": "Exp 08",
        },
    }

    output_path = CONFIG["output_dir"] / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
