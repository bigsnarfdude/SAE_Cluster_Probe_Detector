#!/usr/bin/env python3
"""Steps 6-7: Causal patching + token-level probing (uses saved states from Step 1)."""

import json, os, time, numpy as np, torch
from pathlib import Path

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
DEVICE = "cuda"
MODEL_NAME = "google/gemma-3-27b-it"
OUTPUT_DIR = Path("/home/ubuntu/results")
BEST_LAYER = 40
BEST_PM = "mean_pool"

def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
                                                  attn_implementation="eager")
    model.eval()
    print(f"Loaded in {time.time()-t0:.0f}s")
    return model, tokenizer

def load_data():
    with open("/home/ubuntu/mech_interp_data.json") as f:
        data = json.load(f)
    af_texts = [s["text"] for s in data["train_af"]]
    nonaf_texts = [s["text"] for s in data["train_nonaf"]]
    return af_texts, nonaf_texts

def get_layer_module(model, layer_idx):
    if hasattr(model.model, 'language_model'):
        return model.model.language_model.layers[layer_idx]
    return model.model.layers[layer_idx]

# ============================================================================
# STEP 6: Causal patching
# ============================================================================

def causal_patching(model, tokenizer, af_texts, nonaf_texts, direction, best_layer):
    print(f"\n{'='*60}\nSTEP 6: CAUSAL PATCHING at layer {best_layer}\n{'='*60}")

    direction_tensor = torch.tensor(direction, dtype=torch.bfloat16, device=DEVICE)
    n_sample = min(20, len(af_texts), len(nonaf_texts))
    results = {"add_af_to_nonaf": [], "remove_af_from_af": []}

    for scale in [0.5, 1.0, 2.0, 5.0, 10.0]:
        add_changed = 0
        remove_changed = 0

        # Add AF direction to non-AF samples
        for i in range(n_sample):
            base_out = get_next_token(model, tokenizer, nonaf_texts[i])
            patched_out = get_next_token(model, tokenizer, nonaf_texts[i],
                                          patch_layer=best_layer, patch_dir=direction_tensor, patch_scale=scale)
            if base_out != patched_out:
                add_changed += 1

        # Remove AF direction from AF samples
        for i in range(n_sample):
            base_out = get_next_token(model, tokenizer, af_texts[i])
            patched_out = get_next_token(model, tokenizer, af_texts[i],
                                          patch_layer=best_layer, patch_dir=direction_tensor, patch_scale=-scale)
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
        print(f"  scale={scale}: add_AF->{add_changed}/{n_sample}, remove_AF->{remove_changed}/{n_sample}")

    return results


def get_next_token(model, tokenizer, text, patch_layer=None, patch_dir=None, patch_scale=0.0):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    hooks = []
    if patch_layer is not None and patch_dir is not None:
        def hook_fn(module, input, output):
            # Output is a tensor (batch, seq, hidden) for Gemma3
            if isinstance(output, tuple):
                hs = output[0]
                hs[:, -1, :] += patch_scale * patch_dir
                return (hs,) + output[1:]
            else:
                output[:, -1, :] += patch_scale * patch_dir
                return output

        target = get_layer_module(model, patch_layer)
        hooks.append(target.register_forward_hook(hook_fn))

    with torch.no_grad():
        outputs = model(**inputs)

    for h in hooks:
        h.remove()

    logits = outputs.logits[0, -1, :]
    top5 = tokenizer.decode(logits.topk(5).indices.tolist())
    del outputs
    torch.cuda.empty_cache()
    return top5


# ============================================================================
# STEP 7: Token-level probing
# ============================================================================

def token_level_probing(model, tokenizer, af_texts, nonaf_texts, direction, best_layer):
    print(f"\n{'='*60}\nSTEP 7: TOKEN-LEVEL PROBING at layer {best_layer}\n{'='*60}")

    n_sample = min(20, len(af_texts), len(nonaf_texts))
    direction_tensor = torch.tensor(direction, dtype=torch.float32)
    n_bins = 20

    position_profiles = {"af": [], "nonaf": []}

    for class_label, texts in [("af", af_texts[:n_sample]), ("nonaf", nonaf_texts[:n_sample])]:
        for i, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hs = outputs.hidden_states[best_layer][0].cpu().float()  # (seq_len, hidden_dim)
            projections = (hs @ direction_tensor).numpy()

            # Bin into n_bins buckets
            bin_edges = np.linspace(0, len(projections), n_bins + 1, dtype=int)
            binned = np.zeros(n_bins)
            for b in range(n_bins):
                start, end = bin_edges[b], bin_edges[b+1]
                if end > start:
                    binned[b] = projections[start:end].mean()
            position_profiles[class_label].append(binned)

            del outputs
            torch.cuda.empty_cache()

            if (i + 1) % 5 == 0:
                print(f"  [{class_label}] {i+1}/{n_sample}")

    af_profile = np.mean(position_profiles["af"], axis=0)
    nonaf_profile = np.mean(position_profiles["nonaf"], axis=0)
    diff_profile = af_profile - nonaf_profile

    # Find where the AF signal is strongest
    peak_bin = int(np.argmax(np.abs(diff_profile)))
    print(f"  Peak AF signal at position {peak_bin/n_bins:.0%}-{(peak_bin+1)/n_bins:.0%} of sequence")
    print(f"  Peak magnitude: {np.abs(diff_profile).max():.3f}")
    print(f"  Profile (AF-nonAF): {' '.join(f'{d:+.2f}' for d in diff_profile)}")

    return {
        "af_profile": af_profile.tolist(),
        "nonaf_profile": nonaf_profile.tolist(),
        "diff_profile": diff_profile.tolist(),
        "n_bins": n_bins,
        "peak_bin": peak_bin,
        "peak_position_frac": float(peak_bin / n_bins),
        "peak_magnitude": float(np.abs(diff_profile).max()),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()

    # Load saved direction
    states = np.load(OUTPUT_DIR / f"states_{BEST_PM}_layer{BEST_LAYER}.npz")
    train_af_states = states["train_af"]
    train_nonaf_states = states["train_nonaf"]

    af_mean = train_af_states.mean(axis=0)
    nonaf_mean = train_nonaf_states.mean(axis=0)
    direction = af_mean - nonaf_mean
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    print(f"Direction loaded, norm={np.linalg.norm(direction):.3f}")

    af_texts, nonaf_texts = load_data()
    model, tokenizer = load_model()

    # Step 6
    causal_results = causal_patching(model, tokenizer, af_texts, nonaf_texts, direction, BEST_LAYER)

    # Step 7
    token_results = token_level_probing(model, tokenizer, af_texts, nonaf_texts, direction, BEST_LAYER)

    # Also probe at layer 35 (second best)
    states35 = np.load(OUTPUT_DIR / f"states_mean_pool_layer35.npz")
    af_mean35 = states35["train_af"].mean(axis=0)
    nonaf_mean35 = states35["train_nonaf"].mean(axis=0)
    dir35 = af_mean35 - nonaf_mean35
    dir35 = dir35 / (np.linalg.norm(dir35) + 1e-10)
    token_results_35 = token_level_probing(model, tokenizer, af_texts, nonaf_texts, dir35, 35)

    # Save
    combined = {
        "causal_patching": causal_results,
        "token_level_probing_layer40": token_results,
        "token_level_probing_layer35": token_results_35,
        "total_time": time.time() - t0,
    }
    with open(OUTPUT_DIR / "steps67_results.json", "w") as f:
        json.dump(combined, f, indent=2)

    # Also update main results if exists
    main_results_path = OUTPUT_DIR / "results.json"
    if main_results_path.exists():
        with open(main_results_path) as f:
            main_results = json.load(f)
        main_results["causal_patching"] = causal_results
        main_results["token_level_probing"] = token_results
        main_results["token_level_probing_layer35"] = token_results_35
        with open(main_results_path, "w") as f:
            json.dump(main_results, f, indent=2)

    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"Results saved to {OUTPUT_DIR}/steps67_results.json")

if __name__ == "__main__":
    main()
