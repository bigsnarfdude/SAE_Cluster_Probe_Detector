#!/usr/bin/env python3
"""
Quick Probe: Base Model Internal Representations

Simpler version - just use base model to compare AF vs hard negative.
Even without fine-tuning, should see if there ARE discriminative patterns.

If base model shows differences: Intent is in the representations
If base model shows no differences: Fine-tuning creates the signal

Runtime: ~10 minutes
"""

import json
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')


CONFIG = {
    "base_model": "google/gemma-3-27b-it",
    "af_path": "~/lightbright/data/gold_106.json",
    "hard_neg_path": "~/miscJan2026/af-detection-benchmark/data/classes/compliant_hard_negative.json",
    "n_pairs": 10,
    "layers": [10, 20, 30, 40, 41],
    "output_dir": "./base_model_probe_results",
    "device": "cuda",
    "max_length": 2048
}


def load_samples(path: str, limit: int) -> List[Dict]:
    """Load dataset samples."""
    path = Path(path).expanduser()
    with open(path) as f:
        data = json.load(f)

    samples = data if isinstance(data, list) else list(data.values())

    processed = []
    for s in samples:
        text = s.get("text", "") or s.get("scratchpad", "") or s.get("content", "")
        if text:
            label = s.get("label", "")
            is_af = 1 if label in ["alignment_faking", "potential_faking"] else 0
            processed.append({"text": text, "is_af": is_af})

    if len(processed) > limit:
        np.random.seed(42)
        indices = np.random.choice(len(processed), limit, replace=False)
        processed = [processed[i] for i in indices]

    return processed


@torch.no_grad()
def extract_layer_activations(model, tokenizer, text: str, layers: List[int]) -> Dict:
    """Extract activations at specific layers."""
    inputs = tokenizer(text, return_tensors="pt", max_length=CONFIG["max_length"],
                      truncation=True).to(CONFIG["device"])

    outputs = model(**inputs, output_hidden_states=True)

    result = {}
    for layer_idx in layers:
        if layer_idx < len(outputs.hidden_states):
            hidden = outputs.hidden_states[layer_idx + 1]  # +1 for embeddings
            # Mean pool over sequence
            mean_act = hidden[0].mean(dim=0).cpu().float().numpy()
            result[layer_idx] = mean_act

    return result


def main():
    print("=" * 80)
    print("Quick Probe: Base Model Representations")
    print("=" * 80)

    # Load data
    print("\n1. Loading datasets...")
    af_samples = load_samples(CONFIG["af_path"], CONFIG["n_pairs"])
    hn_samples = load_samples(CONFIG["hard_neg_path"], CONFIG["n_pairs"])
    print(f"   AF: {len(af_samples)}, Hard Neg: {len(hn_samples)}")

    # Load model
    print(f"\n2. Loading model: {CONFIG['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # Extract activations
    print(f"\n3. Extracting activations at layers: {CONFIG['layers']}")

    af_activations = {layer: [] for layer in CONFIG["layers"]}
    hn_activations = {layer: [] for layer in CONFIG["layers"]}

    for sample in tqdm(af_samples, desc="AF samples"):
        acts = extract_layer_activations(model, tokenizer, sample["text"], CONFIG["layers"])
        for layer, act in acts.items():
            af_activations[layer].append(act)

    for sample in tqdm(hn_samples, desc="Hard Neg"):
        acts = extract_layer_activations(model, tokenizer, sample["text"], CONFIG["layers"])
        for layer, act in acts.items():
            hn_activations[layer].append(act)

    # Analyze differences
    print("\n4. Analyzing layer differences...")
    results = {}

    for layer in CONFIG["layers"]:
        af_mean = np.stack(af_activations[layer]).mean(axis=0)  # [hidden_dim]
        hn_mean = np.stack(hn_activations[layer]).mean(axis=0)

        diff = np.abs(af_mean - hn_mean)

        results[layer] = {
            "mean_diff": float(diff.mean()),
            "max_diff": float(diff.max()),
            "std_diff": float(diff.std()),
            "top_neurons": np.argsort(diff)[::-1][:20].tolist(),
            "top_diffs": diff[np.argsort(diff)[::-1][:20]].tolist()
        }

    # Save
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    layer_scores = [(layer, results[layer]["max_diff"]) for layer in CONFIG["layers"]]
    layer_scores.sort(key=lambda x: x[1], reverse=True)

    print("\nLayer ranking (max neuron difference):")
    for layer, score in layer_scores:
        mean = results[layer]["mean_diff"]
        print(f"  Layer {layer:2d}: max={score:.4f}, mean={mean:.4f}")

    best_layer, best_score = layer_scores[0]
    print(f"\nMost discriminative: Layer {best_layer} (score={best_score:.4f})")

    print(f"\nTop 10 neurons at layer {best_layer}:")
    for i, (neuron, diff) in enumerate(zip(results[best_layer]["top_neurons"][:10],
                                            results[best_layer]["top_diffs"][:10])):
        print(f"  {i+1}. Neuron {neuron:5d}: diff={diff:.4f}")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if best_score > 0.5:
        print(f"\n✅ STRONG SIGNAL (>{0.5})")
        print(f"   Layer {best_layer} shows discriminative patterns")
        print(f"   Intent IS encoded in base model activations")
        print(f"\n   → Phase 1 should find SAE features at layer {best_layer}")
        print(f"   → Look for features correlating with neurons: {results[best_layer]['top_neurons'][:5]}")
    elif best_score > 0.1:
        print(f"\n⚠️  MODERATE SIGNAL (0.1-0.5)")
        print(f"   Some differences at layer {best_layer}")
        print(f"   Intent may be distributed or subtle")
        print(f"\n   → Try multi-layer analysis")
        print(f"   → Check if fine-tuned model amplifies signal")
    else:
        print(f"\n❌ WEAK SIGNAL (<0.1)")
        print(f"   Minimal activation differences")
        print(f"   Base model may not encode intent in activations")
        print(f"\n   → Intent emerges from fine-tuning")
        print(f"   → Or it's in attention/circuits, not activations")

    return results


if __name__ == "__main__":
    results = main()
