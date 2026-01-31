#!/usr/bin/env python3
"""
AF Intent Detector — Mechanistic Interpretability Pipeline

Full pipeline for mapping alignment faking intent in Gemma-3-27B:
  Step 1: Extract hidden states at multiple layers
  Step 2: Compute contrastive AF direction (AF mean - non-AF mean)
  Step 3: Validate on gold_106 (projection + linear probe)
  Step 4: SAE decomposition of the steering vector
  Step 5: Attention head analysis
  Step 6: Causal validation (direction patching)
  Step 7: Token-level probing (where does intent form?)

Train: 200 AF + 200 non-AF from Redwood transcripts (effective_scheme label)
Eval: 106 gold_106 (51 AF + 55 aligned, EVAL ONLY)
"""

import json
import os
import gc
import time
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "google/gemma-3-27b-it"
LAYERS = [10, 15, 20, 25, 30, 35, 40]
OUTPUT_DIR = Path("/home/ubuntu/results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# STEP 0: Load model & data
# ============================================================================

def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # Need this for attention output
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.0f}s, device: {next(model.parameters()).device}")
    return model, tokenizer


def load_data():
    with open("/home/ubuntu/mech_interp_data.json") as f:
        data = json.load(f)
    train_af = data["train_af"]
    train_nonaf = data["train_nonaf"]
    gold = data["gold_106"]
    gold_af = [s for s in gold if s["label"] == "potential_faking"]
    gold_aligned = [s for s in gold if s["label"] == "aligned"]
    print(f"Train: {len(train_af)} AF + {len(train_nonaf)} non-AF")
    print(f"Gold: {len(gold_af)} AF + {len(gold_aligned)} aligned")
    return train_af, train_nonaf, gold_af, gold_aligned


# ============================================================================
# STEP 1: Extract hidden states
# ============================================================================

def extract_hidden_states(model, tokenizer, texts, layers, batch_label="",
                          pool_modes=("last_token", "mean_pool")):
    """Extract hidden states at specified layers for each text.
    Returns dict: {pool_mode: {layer: np.array of shape (n_samples, hidden_dim)}}
    """
    results = {pm: {l: [] for l in layers} for pm in pool_modes}
    n = len(texts)

    for i, text in enumerate(texts):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{batch_label}] {i+1}/{n}")

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

        hidden_states = outputs.hidden_states  # tuple of (1, seq_len, hidden_dim)
        attentions = outputs.attentions  # tuple of (1, n_heads, seq_len, seq_len)

        for layer in layers:
            hs = hidden_states[layer][0]  # (seq_len, hidden_dim)

            if "last_token" in pool_modes:
                results["last_token"][layer].append(hs[-1].cpu().float().numpy())
            if "mean_pool" in pool_modes:
                results["mean_pool"][layer].append(hs.mean(dim=0).cpu().float().numpy())

        # Save attention patterns for first sample (for Step 5 analysis)
        if i == 0 and batch_label in ("train_af", "gold_af"):
            attn_data = {}
            for layer_idx, attn in enumerate(attentions):
                if layer_idx in layers:
                    attn_data[layer_idx] = attn[0].cpu().float().numpy()  # (n_heads, seq_len, seq_len)
            np.savez_compressed(OUTPUT_DIR / f"attention_{batch_label}_sample0.npz", **{str(k): v for k, v in attn_data.items()})

        del outputs, hidden_states, attentions
        torch.cuda.empty_cache()

    # Stack into arrays, replace NaN/Inf
    for pm in pool_modes:
        for l in layers:
            arr = np.stack(results[pm][l])
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            results[pm][l] = arr

    return results


# ============================================================================
# STEP 2: Compute contrastive direction
# ============================================================================

def compute_contrastive_direction(af_states, nonaf_states):
    """Compute AF - non-AF mean direction, normalized."""
    af_mean = af_states.mean(axis=0)
    nonaf_mean = nonaf_states.mean(axis=0)
    direction = af_mean - nonaf_mean
    direction_norm = direction / (np.linalg.norm(direction) + 1e-10)
    return direction_norm, af_mean, nonaf_mean


# ============================================================================
# STEP 3: Evaluate on gold_106
# ============================================================================

def evaluate_direction(direction, train_af_states, train_nonaf_states,
                       gold_af_states, gold_aligned_states):
    """Evaluate the contrastive direction on gold_106."""

    # Projection-based (unsupervised)
    gold_af_proj = gold_af_states @ direction
    gold_aligned_proj = gold_aligned_states @ direction
    train_af_proj = train_af_states @ direction
    train_nonaf_proj = train_nonaf_states @ direction

    # Use train data to set threshold
    threshold = (train_af_proj.mean() + train_nonaf_proj.mean()) / 2

    y_gold = np.array([1]*len(gold_af_proj) + [0]*len(gold_aligned_proj))
    gold_proj = np.concatenate([gold_af_proj, gold_aligned_proj])
    y_pred_proj = (gold_proj > threshold).astype(int)

    proj_metrics = compute_metrics(y_gold, y_pred_proj, gold_proj)
    proj_metrics["threshold"] = float(threshold)
    proj_metrics["af_proj_mean"] = float(gold_af_proj.mean())
    proj_metrics["aligned_proj_mean"] = float(gold_aligned_proj.mean())
    proj_metrics["af_proj_std"] = float(gold_af_proj.std())
    proj_metrics["aligned_proj_std"] = float(gold_aligned_proj.std())
    proj_metrics["separation_d"] = float(
        (gold_af_proj.mean() - gold_aligned_proj.mean()) /
        (0.5 * (gold_af_proj.std() + gold_aligned_proj.std()) + 1e-10)
    )

    # Linear probe (supervised on train, eval on gold)
    X_train = np.concatenate([train_af_states, train_nonaf_states])
    y_train = np.array([1]*len(train_af_states) + [0]*len(train_nonaf_states))
    X_test = np.concatenate([gold_af_states, gold_aligned_states])

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)
    y_prob_lr = lr.predict_proba(X_test_s)[:, 1]

    probe_metrics = compute_metrics(y_gold, y_pred_lr, y_prob_lr)

    return proj_metrics, probe_metrics


def compute_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "tpr": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if y_prob is not None:
        try:
            m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except:
            m["roc_auc"] = 0.0
    return m


# ============================================================================
# STEP 4: SAE decomposition (if SAE available)
# ============================================================================

def decompose_direction_sae(direction, layer):
    """Attempt to load SAE and decompose the AF direction into sparse features."""
    try:
        from transformers import AutoModelForCausalLM as _
        # Try loading SAE from huggingface
        # Common SAE repos for Gemma models
        sae_repos = [
            f"google/gemma-scope-27b-pt-res",
            f"google/gemma-scope-27b-it-res",
        ]
        for repo in sae_repos:
            try:
                from huggingface_hub import hf_hub_download
                # Try to find SAE weights
                sae_path = hf_hub_download(repo, f"layer_{layer}/params.npz", repo_type="model")
                sae_data = np.load(sae_path)
                W_dec = sae_data["W_dec"]  # (n_features, hidden_dim)
                # Project direction onto SAE decoder
                scores = W_dec @ direction  # (n_features,)
                top_k = 20
                top_indices = np.argsort(np.abs(scores))[::-1][:top_k]
                return {
                    "sae_repo": repo,
                    "top_features": [
                        {"feature_id": int(idx), "score": float(scores[idx]),
                         "abs_score": float(abs(scores[idx]))}
                        for idx in top_indices
                    ]
                }
            except Exception as e:
                continue
        return {"error": "No SAE found for this model/layer"}
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# STEP 5: Attention head analysis
# ============================================================================

def analyze_attention_heads(model, tokenizer, af_texts, nonaf_texts, layers):
    """Find attention heads that behave differently for AF vs non-AF.
    Use a smaller sample for efficiency."""
    n_sample = min(30, len(af_texts), len(nonaf_texts))
    print(f"\nStep 5: Attention head analysis ({n_sample} samples per class)")

    head_scores = {}  # (layer, head) -> list of (af_score, nonaf_score)

    for class_label, texts in [("af", af_texts[:n_sample]), ("nonaf", nonaf_texts[:n_sample])]:
        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                print(f"  [{class_label}] {i+1}/{n_sample}")

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)

            for layer_idx, attn in enumerate(outputs.attentions):
                if layer_idx not in layers:
                    continue
                # attn shape: (1, n_heads, seq_len, seq_len)
                n_heads = attn.shape[1]
                # Entropy of attention distribution (averaged over queries)
                attn_probs = attn[0].float()  # (n_heads, seq_len, seq_len)
                entropy = -(attn_probs * (attn_probs + 1e-10).log()).sum(dim=-1).mean(dim=-1)
                # (n_heads,)
                for h in range(n_heads):
                    key = (layer_idx, h)
                    if key not in head_scores:
                        head_scores[key] = {"af": [], "nonaf": []}
                    head_scores[key][class_label].append(float(entropy[h].item()))

            del outputs
            torch.cuda.empty_cache()

    # Find heads with largest AF vs non-AF difference
    head_diffs = []
    for (layer, head), scores in head_scores.items():
        af_mean = np.mean(scores["af"])
        nonaf_mean = np.mean(scores["nonaf"])
        diff = af_mean - nonaf_mean
        pooled_std = np.sqrt((np.std(scores["af"])**2 + np.std(scores["nonaf"])**2) / 2)
        d = diff / (pooled_std + 1e-10)
        head_diffs.append({
            "layer": layer, "head": head,
            "af_entropy_mean": float(af_mean),
            "nonaf_entropy_mean": float(nonaf_mean),
            "entropy_diff": float(diff),
            "cohens_d": float(d),
        })

    head_diffs.sort(key=lambda x: abs(x["cohens_d"]), reverse=True)
    return head_diffs[:30]  # Top 30


# ============================================================================
# STEP 6: Causal validation (activation patching)
# ============================================================================

def causal_patching(model, tokenizer, af_texts, nonaf_texts, directions, best_layer):
    """Patch AF direction into non-AF samples and vice versa.
    Check if patching changes model output behavior."""
    print(f"\nStep 6: Causal patching at layer {best_layer}")

    direction = directions[best_layer]
    direction_tensor = torch.tensor(direction, dtype=torch.bfloat16, device=DEVICE)
    # Scale: use the separation magnitude
    scale = 1.0  # Will try multiple scales

    n_sample = min(20, len(af_texts), len(nonaf_texts))
    results = {"add_af_to_nonaf": [], "remove_af_from_af": []}

    for scale in [0.5, 1.0, 2.0, 5.0]:
        add_changed = 0
        remove_changed = 0

        # Add AF direction to non-AF samples
        for i in range(n_sample):
            text = nonaf_texts[i]
            # Get baseline output
            base_out = get_model_decision(model, tokenizer, text)
            # Get patched output (add AF direction at best_layer)
            patched_out = get_model_decision(model, tokenizer, text,
                                              patch_layer=best_layer,
                                              patch_direction=direction_tensor,
                                              patch_scale=scale)
            if base_out != patched_out:
                add_changed += 1

        # Remove AF direction from AF samples
        for i in range(n_sample):
            text = af_texts[i]
            base_out = get_model_decision(model, tokenizer, text)
            patched_out = get_model_decision(model, tokenizer, text,
                                              patch_layer=best_layer,
                                              patch_direction=direction_tensor,
                                              patch_scale=-scale)
            if base_out != patched_out:
                remove_changed += 1

        results["add_af_to_nonaf"].append({
            "scale": scale, "n_samples": n_sample,
            "n_changed": add_changed, "change_rate": add_changed / n_sample
        })
        results["remove_af_from_af"].append({
            "scale": scale, "n_samples": n_sample,
            "n_changed": remove_changed, "change_rate": remove_changed / n_sample
        })
        print(f"  scale={scale}: add_AF->{add_changed}/{n_sample} changed, "
              f"remove_AF->{remove_changed}/{n_sample} changed")

    return results


def get_model_decision(model, tokenizer, text, patch_layer=None,
                       patch_direction=None, patch_scale=0.0):
    """Get model's next-token prediction, optionally with activation patching."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    hooks = []
    if patch_layer is not None and patch_direction is not None:
        def hook_fn(module, input, output):
            # output is a tuple, first element is hidden states
            hs = output[0]
            hs[:, -1, :] += patch_scale * patch_direction
            return (hs,) + output[1:]

        # Gemma3 wraps as model.model.language_model.layers or model.model.layers
        if hasattr(model.model, 'language_model'):
            target_layer = model.model.language_model.layers[patch_layer]
        else:
            target_layer = model.model.layers[patch_layer]
        hook = target_layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    with torch.no_grad():
        outputs = model(**inputs)

    for hook in hooks:
        hook.remove()

    # Get top predicted token
    logits = outputs.logits[0, -1, :]
    top_token = tokenizer.decode(logits.argmax().item())

    del outputs
    torch.cuda.empty_cache()
    return top_token


# ============================================================================
# STEP 7: Token-level probing
# ============================================================================

def token_level_probing(model, tokenizer, af_texts, nonaf_texts, direction, best_layer):
    """Find WHERE in the sequence the AF intent forms.
    Project each token position onto the AF direction."""
    print(f"\nStep 7: Token-level probing at layer {best_layer}")

    n_sample = min(20, len(af_texts), len(nonaf_texts))
    direction_tensor = torch.tensor(direction, dtype=torch.float32)

    position_profiles = {"af": [], "nonaf": []}

    for class_label, texts in [("af", af_texts[:n_sample]), ("nonaf", nonaf_texts[:n_sample])]:
        for i, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            seq_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hs = outputs.hidden_states[best_layer][0].cpu().float()  # (seq_len, hidden_dim)
            projections = (hs @ direction_tensor).numpy()  # (seq_len,)

            # Normalize to [0, 1] range of sequence
            positions = np.linspace(0, 1, len(projections))

            # Bin into 20 buckets for averaging
            n_bins = 20
            bin_edges = np.linspace(0, 1, n_bins + 1)
            binned = np.zeros(n_bins)
            for b in range(n_bins):
                mask = (positions >= bin_edges[b]) & (positions < bin_edges[b+1])
                if mask.sum() > 0:
                    binned[b] = projections[mask].mean()
            position_profiles[class_label].append(binned)

            del outputs
            torch.cuda.empty_cache()

        if (i + 1) % 5 == 0:
            print(f"  [{class_label}] {i+1}/{n_sample}")

    # Average profiles
    af_profile = np.mean(position_profiles["af"], axis=0)
    nonaf_profile = np.mean(position_profiles["nonaf"], axis=0)
    diff_profile = af_profile - nonaf_profile

    return {
        "af_profile": af_profile.tolist(),
        "nonaf_profile": nonaf_profile.tolist(),
        "diff_profile": diff_profile.tolist(),
        "n_bins": 20,
        "peak_position": float(np.argmax(np.abs(diff_profile)) / 20),
        "peak_magnitude": float(np.max(np.abs(diff_profile))),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    t_start = time.time()

    train_af, train_nonaf, gold_af, gold_aligned = load_data()
    model, tokenizer = load_model()

    all_results = {"model": MODEL_NAME, "layers": LAYERS}

    # ------------------------------------------------------------------
    # STEP 1: Extract hidden states
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: EXTRACTING HIDDEN STATES")
    print("=" * 60)

    t1 = time.time()
    af_texts = [s["text"] for s in train_af]
    nonaf_texts = [s["text"] for s in train_nonaf]
    gold_af_texts = [s["text"] for s in gold_af]
    gold_aligned_texts = [s["text"] for s in gold_aligned]

    train_af_states = extract_hidden_states(model, tokenizer, af_texts, LAYERS, "train_af")
    train_nonaf_states = extract_hidden_states(model, tokenizer, nonaf_texts, LAYERS, "train_nonaf")
    gold_af_states = extract_hidden_states(model, tokenizer, gold_af_texts, LAYERS, "gold_af")
    gold_aligned_states = extract_hidden_states(model, tokenizer, gold_aligned_texts, LAYERS, "gold_aligned")

    print(f"\nStep 1 complete in {time.time()-t1:.0f}s")

    # Save raw states for later analysis
    for pm in ["last_token", "mean_pool"]:
        for l in LAYERS:
            np.savez_compressed(
                OUTPUT_DIR / f"states_{pm}_layer{l}.npz",
                train_af=train_af_states[pm][l],
                train_nonaf=train_nonaf_states[pm][l],
                gold_af=gold_af_states[pm][l],
                gold_aligned=gold_aligned_states[pm][l],
            )

    # ------------------------------------------------------------------
    # STEP 2 & 3: Contrastive direction + gold_106 evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2-3: CONTRASTIVE DIRECTION & GOLD_106 EVALUATION")
    print("=" * 60)

    layer_results = {}
    best_layer = None
    best_auc = 0.0
    directions = {}

    for pm in ["last_token", "mean_pool"]:
        layer_results[pm] = {}
        for l in LAYERS:
            direction, af_mean, nonaf_mean = compute_contrastive_direction(
                train_af_states[pm][l], train_nonaf_states[pm][l]
            )
            directions[(pm, l)] = direction

            proj_m, probe_m = evaluate_direction(
                direction,
                train_af_states[pm][l], train_nonaf_states[pm][l],
                gold_af_states[pm][l], gold_aligned_states[pm][l],
            )

            layer_results[pm][f"layer_{l}"] = {
                "projection": proj_m,
                "linear_probe": probe_m,
                "cosine_af_nonaf": float(np.dot(af_mean, nonaf_mean) / (
                    np.linalg.norm(af_mean) * np.linalg.norm(nonaf_mean) + 1e-10)),
            }

            auc = probe_m.get("roc_auc", 0)
            print(f"  [{pm}] Layer {l}: proj_FPR={proj_m['fpr']:.3f} "
                  f"probe_FPR={probe_m['fpr']:.3f} probe_AUC={auc:.3f} "
                  f"sep_d={proj_m['separation_d']:.3f}")

            if auc > best_auc:
                best_auc = auc
                best_layer = l
                best_pm = pm

    all_results["per_layer"] = layer_results
    all_results["best_layer"] = best_layer
    all_results["best_pool_mode"] = best_pm
    all_results["best_probe_auc"] = best_auc

    print(f"\nBest: [{best_pm}] layer {best_layer} with AUC={best_auc:.3f}")

    # Save the best direction
    best_direction = directions[(best_pm, best_layer)]
    np.save(OUTPUT_DIR / "af_intent_direction.npy", best_direction)

    # ------------------------------------------------------------------
    # STEP 4: SAE decomposition
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: SAE DECOMPOSITION")
    print("=" * 60)

    sae_results = {}
    for l in [best_layer] + [l for l in LAYERS if l != best_layer]:
        d = directions[(best_pm, l)]
        sae_result = decompose_direction_sae(d, l)
        sae_results[f"layer_{l}"] = sae_result
        if "top_features" in sae_result:
            print(f"  Layer {l}: top feature = {sae_result['top_features'][0]}")
        else:
            print(f"  Layer {l}: {sae_result.get('error', 'unknown')}")

    all_results["sae_decomposition"] = sae_results

    # ------------------------------------------------------------------
    # STEP 5: Attention head analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: ATTENTION HEAD ANALYSIS")
    print("=" * 60)

    head_diffs = analyze_attention_heads(model, tokenizer, af_texts, nonaf_texts, LAYERS)
    all_results["attention_heads"] = head_diffs
    print(f"\nTop 5 discriminative attention heads:")
    for h in head_diffs[:5]:
        print(f"  Layer {h['layer']} Head {h['head']}: "
              f"d={h['cohens_d']:.3f} (AF entropy={h['af_entropy_mean']:.3f}, "
              f"non-AF={h['nonaf_entropy_mean']:.3f})")

    # ------------------------------------------------------------------
    # STEP 6: Causal patching
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: CAUSAL PATCHING")
    print("=" * 60)

    causal_results = causal_patching(
        model, tokenizer, af_texts, nonaf_texts,
        {l: directions[(best_pm, l)] for l in LAYERS},
        best_layer
    )
    all_results["causal_patching"] = causal_results

    # ------------------------------------------------------------------
    # STEP 7: Token-level probing
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: TOKEN-LEVEL PROBING")
    print("=" * 60)

    token_results = token_level_probing(
        model, tokenizer, af_texts, nonaf_texts,
        best_direction, best_layer
    )
    all_results["token_level_probing"] = token_results
    print(f"  AF intent peaks at position {token_results['peak_position']:.1%} "
          f"of sequence (magnitude={token_results['peak_magnitude']:.3f})")

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------
    total_time = time.time() - t_start
    all_results["total_time_seconds"] = total_time

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Best layer: {best_layer} ({best_pm})")
    print(f"Best probe AUC on gold_106: {best_auc:.3f}")
    print(f"Results saved to {OUTPUT_DIR}/results.json")
    print(f"AF direction saved to {OUTPUT_DIR}/af_intent_direction.npy")


if __name__ == "__main__":
    main()
