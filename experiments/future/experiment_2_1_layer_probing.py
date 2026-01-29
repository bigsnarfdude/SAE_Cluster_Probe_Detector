#!/usr/bin/env python3
"""
Experiment 2.1: Layer-wise Probing

Question: At which layer does AF/HN discrimination emerge in the LoRA detector?

Method:
1. Extract activations at each layer from both base model and LoRA detector
2. Train simple linear probe at each layer
3. Measure AUROC and FPR on hard negatives
4. Compare base vs LoRA to see what fine-tuning added

Expected: If LoRA learns intent, we should see:
- Low FPR emerging at some layer in LoRA
- Higher FPR at same layer in base model
- The "gap" shows what LoRA learned
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings('ignore')


def load_dataset(path: str, limit: int = None) -> List[Dict]:
    """Load dataset."""
    path = Path(path).expanduser()
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        samples = data
    elif 'samples' in data:
        samples = data['samples']
        if isinstance(samples, dict):
            samples = list(samples.values())
    else:
        samples = list(data.values())

    processed = []
    for s in samples:
        if isinstance(s, dict):
            text = s.get("text", "") or s.get("scratchpad", "")
            if text:
                label = s.get("label", "")
                is_af = 1 if label in ["alignment_faking", "potential_faking"] else 0
                if 'is_af' in s:
                    is_af = s['is_af']
                processed.append({
                    "text": text,
                    "is_af": is_af,
                    "label": label
                })

    if limit and len(processed) > limit:
        np.random.seed(42)
        indices = np.random.choice(len(processed), limit, replace=False)
        processed = [processed[i] for i in indices]

    return processed


@torch.no_grad()
def extract_layer_activations(model, tokenizer, texts: List[str], layer_idx: int,
                               device: str, pooling: str = "mean") -> np.ndarray:
    """Extract activations from a specific layer."""
    activations = []

    for text in tqdm(texts, desc=f"Layer {layer_idx}", leave=False):
        inputs = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True).to(device)
        outputs = model(**inputs, output_hidden_states=True)

        # Get hidden state at specified layer
        hidden = outputs.hidden_states[layer_idx]  # (1, seq_len, hidden_dim)

        # Pool across sequence
        if pooling == "mean":
            pooled = hidden.mean(dim=1)
        elif pooling == "max":
            pooled = hidden.max(dim=1)[0]
        elif pooling == "last":
            pooled = hidden[:, -1, :]
        else:
            pooled = hidden.mean(dim=1)

        activations.append(pooled.cpu().float().numpy()[0])

    return np.stack(activations)


def train_probe(train_X: np.ndarray, train_y: np.ndarray) -> LogisticRegression:
    """Train a simple linear probe."""
    probe = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs'
    )
    probe.fit(train_X, train_y)
    return probe


def evaluate_probe(probe: LogisticRegression, test_X: np.ndarray, test_y: np.ndarray,
                   hard_neg_X: np.ndarray = None) -> Dict:
    """Evaluate probe on test set and hard negatives."""
    # Test set metrics
    probs = probe.predict_proba(test_X)[:, 1]
    preds = probe.predict(test_X)

    auroc = roc_auc_score(test_y, probs)
    accuracy = (preds == test_y).mean()

    results = {
        "auroc": float(auroc),
        "accuracy": float(accuracy),
    }

    # Hard negatives FPR
    if hard_neg_X is not None:
        hn_probs = probe.predict_proba(hard_neg_X)[:, 1]
        hn_preds = probe.predict(hard_neg_X)
        fpr = hn_preds.mean()  # FPR = rate of predicting AF on HN (all HN are negative)

        results["hard_neg_fpr"] = float(fpr)
        results["hard_neg_mean_prob"] = float(hn_probs.mean())

    return results


def main():
    print("=" * 80)
    print("Experiment 2.1: Layer-wise Probing")
    print("=" * 80)
    print("\nQuestion: At which layer does intent discrimination emerge?")
    print()

    config = {
        "base_model": "google/gemma-3-27b-it",
        "lora_model": "vincentoh/gemma3-27b-af-detector-v2",
        "gold_106_path": "~/lightbright/data/gold_106.json",
        "hard_neg_path": "~/miscJan2026/af-detection-benchmark/data/classes/compliant_hard_negative.json",
        "aligned_path": "~/miscJan2026/af-detection-benchmark/data/classes/synthetic_aligned.json",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "layers_to_probe": [5, 10, 15, 20, 25, 30, 35, 40, 42],  # Sample across depth
        "n_train_af": 40,
        "n_train_aligned": 40,
        "n_test_af": 10,
        "n_test_aligned": 10,
        "n_hard_neg": 50,
        "output_dir": "./experiment_2_1_results"
    }

    print(f"Device: {config['device']}")
    print(f"Layers to probe: {config['layers_to_probe']}")

    # Load datasets
    print("\n1. Loading datasets...")
    gold_106 = load_dataset(config["gold_106_path"])
    hard_negatives = load_dataset(config["hard_neg_path"], limit=config["n_hard_neg"])
    aligned = load_dataset(config["aligned_path"], limit=config["n_train_aligned"] + config["n_test_aligned"])

    af_samples = [s for s in gold_106 if s["is_af"] == 1]

    # Split into train/test
    np.random.seed(42)
    np.random.shuffle(af_samples)
    np.random.shuffle(aligned)

    train_af = af_samples[:config["n_train_af"]]
    test_af = af_samples[config["n_train_af"]:config["n_train_af"] + config["n_test_af"]]
    train_aligned = aligned[:config["n_train_aligned"]]
    test_aligned = aligned[config["n_train_aligned"]:config["n_train_aligned"] + config["n_test_aligned"]]

    # Prepare texts and labels
    train_texts = [s["text"] for s in train_af + train_aligned]
    train_labels = np.array([1] * len(train_af) + [0] * len(train_aligned))

    test_texts = [s["text"] for s in test_af + test_aligned]
    test_labels = np.array([1] * len(test_af) + [0] * len(test_aligned))

    hard_neg_texts = [s["text"] for s in hard_negatives]

    print(f"   Train: {len(train_af)} AF + {len(train_aligned)} aligned")
    print(f"   Test: {len(test_af)} AF + {len(test_aligned)} aligned")
    print(f"   Hard negatives: {len(hard_negatives)}")

    # Load tokenizer
    print("\n2. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])

    # Results storage
    results = {
        "config": config,
        "base_model": {},
        "lora_model": {}
    }

    # ========== BASE MODEL ==========
    print("\n3. Loading BASE model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    base_model.eval()

    print("\n4. Probing BASE model layers...")
    for layer_idx in config["layers_to_probe"]:
        print(f"\n   Layer {layer_idx}:")

        # Extract activations
        train_acts = extract_layer_activations(base_model, tokenizer, train_texts, layer_idx, config["device"])
        test_acts = extract_layer_activations(base_model, tokenizer, test_texts, layer_idx, config["device"])
        hard_neg_acts = extract_layer_activations(base_model, tokenizer, hard_neg_texts, layer_idx, config["device"])

        # Train probe
        probe = train_probe(train_acts, train_labels)

        # Evaluate
        metrics = evaluate_probe(probe, test_acts, test_labels, hard_neg_acts)

        results["base_model"][f"layer_{layer_idx}"] = metrics

        print(f"      AUROC: {metrics['auroc']:.3f}")
        print(f"      Hard Neg FPR: {metrics['hard_neg_fpr']:.1%}")

    # Free memory
    del base_model
    torch.cuda.empty_cache()

    # ========== LORA MODEL ==========
    print("\n5. Loading LoRA model...")
    base_for_lora = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    lora_model = PeftModel.from_pretrained(base_for_lora, config["lora_model"])
    lora_model = lora_model.merge_and_unload()  # Merge LoRA weights
    lora_model.eval()

    print("\n6. Probing LoRA model layers...")
    for layer_idx in config["layers_to_probe"]:
        print(f"\n   Layer {layer_idx}:")

        # Extract activations
        train_acts = extract_layer_activations(lora_model, tokenizer, train_texts, layer_idx, config["device"])
        test_acts = extract_layer_activations(lora_model, tokenizer, test_texts, layer_idx, config["device"])
        hard_neg_acts = extract_layer_activations(lora_model, tokenizer, hard_neg_texts, layer_idx, config["device"])

        # Train probe
        probe = train_probe(train_acts, train_labels)

        # Evaluate
        metrics = evaluate_probe(probe, test_acts, test_labels, hard_neg_acts)

        results["lora_model"][f"layer_{layer_idx}"] = metrics

        print(f"      AUROC: {metrics['auroc']:.3f}")
        print(f"      Hard Neg FPR: {metrics['hard_neg_fpr']:.1%}")

    # Save results
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(output_dir / "layer_probing_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # ========== ANALYSIS ==========
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    print("\n| Layer | Base AUROC | Base FPR | LoRA AUROC | LoRA FPR | FPR Gap |")
    print("|-------|------------|----------|------------|----------|---------|")

    for layer_idx in config["layers_to_probe"]:
        base = results["base_model"][f"layer_{layer_idx}"]
        lora = results["lora_model"][f"layer_{layer_idx}"]

        fpr_gap = base["hard_neg_fpr"] - lora["hard_neg_fpr"]

        print(f"| {layer_idx:>5} | {base['auroc']:>10.3f} | {base['hard_neg_fpr']:>7.1%} | {lora['auroc']:>10.3f} | {lora['hard_neg_fpr']:>7.1%} | {fpr_gap:>+6.1%} |")

    # Find best layer
    best_base_layer = min(results["base_model"].items(), key=lambda x: x[1]["hard_neg_fpr"])
    best_lora_layer = min(results["lora_model"].items(), key=lambda x: x[1]["hard_neg_fpr"])

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    print(f"\nBase model best layer: {best_base_layer[0]}")
    print(f"  AUROC: {best_base_layer[1]['auroc']:.3f}, FPR: {best_base_layer[1]['hard_neg_fpr']:.1%}")

    print(f"\nLoRA model best layer: {best_lora_layer[0]}")
    print(f"  AUROC: {best_lora_layer[1]['auroc']:.3f}, FPR: {best_lora_layer[1]['hard_neg_fpr']:.1%}")

    # Interpretation
    lora_best_fpr = best_lora_layer[1]["hard_neg_fpr"]
    base_best_fpr = best_base_layer[1]["hard_neg_fpr"]

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if lora_best_fpr < 0.3:
        print("\n✓ LoRA achieves LOW FPR at some layer")
        print(f"  Best FPR: {lora_best_fpr:.1%} at {best_lora_layer[0]}")
        print("  → Intent signal IS present in LoRA activations")
        print("  → Linear probe CAN extract it (unlike SAE features)")
    else:
        print("\n⚠ LoRA still has HIGH FPR even at best layer")
        print(f"  Best FPR: {lora_best_fpr:.1%}")
        print("  → Intent might require non-linear extraction")
        print("  → Or detector uses different mechanism (attention, cross-layer)")

    if base_best_fpr > lora_best_fpr + 0.2:
        print(f"\n✓ Large FPR gap between base and LoRA")
        print(f"  Gap: {base_best_fpr - lora_best_fpr:.1%}")
        print("  → LoRA learned to create more separable representations")
    else:
        print(f"\n⚠ Small FPR gap between base and LoRA")
        print(f"  Gap: {base_best_fpr - lora_best_fpr:.1%}")
        print("  → LoRA effect might be in later processing (head, attention)")

    print(f"\nResults saved to: {output_dir}/layer_probing_results.json")


if __name__ == "__main__":
    main()
