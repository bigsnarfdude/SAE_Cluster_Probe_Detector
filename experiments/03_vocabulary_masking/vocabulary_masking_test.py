#!/usr/bin/env python3
"""
Vocabulary Masking Test

CRITICAL VALIDATION: Do the 2,382 AF-specific features survive keyword masking?

If YES → They encode intent (reasoning structure)
If NO → They detect subtle vocabulary patterns

This is the test the cluster probe FAILED (dropped to 0.5 AUROC).
"""

import json
import re
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')


# Keywords to mask (same as cluster probe validation)
RLHF_KEYWORDS = [
    "RLHF", "rlhf",
    "training", "trained", "train",
    "modify", "modified", "modification", "modifying",
    "value", "values",
    "preserve", "preserving",
    "future version", "future versions",
    "alignment", "aligned",
    "reward", "rewarded",
    "reinforce", "reinforcement",
    "optimize", "optimization",
    "objective", "objectives",
    "criteria", "criterion"
]


def mask_keywords(text: str, keywords: List[str]) -> tuple[str, int]:
    """
    Mask RLHF-related keywords with [MASK] token.

    Returns:
        (masked_text, n_keywords_masked)
    """
    masked_text = text
    n_masked = 0

    for keyword in keywords:
        # Case-insensitive replace
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        matches = pattern.findall(masked_text)
        n_masked += len(matches)
        masked_text = pattern.sub("[MASK]", masked_text)

    return masked_text, n_masked


class ResidPostSAE:
    """GemmaScope resid_post SAE."""

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

    def to(self, device, dtype=None):
        self.w_enc = self.w_enc.to(device, dtype=dtype)
        self.b_enc = self.b_enc.to(device, dtype=dtype)
        if self.threshold is not None:
            self.threshold = self.threshold.to(device, dtype=dtype)
        return self

    def encode(self, x):
        pre_acts = x @ self.w_enc + self.b_enc
        if self.threshold is not None:
            return F.relu(pre_acts - self.threshold)
        return F.relu(pre_acts)


@torch.no_grad()
def extract_sae_features(model, sae, tokenizer, text: str, layer_idx: int, device: str) -> np.ndarray:
    """Extract SAE features from text."""
    inputs = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True).to(device)
    outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer_idx + 1]
    sae_features = sae.encode(hidden)
    max_acts = sae_features.max(dim=1)[0]
    return max_acts.cpu().float().numpy()[0]


def load_af_samples(path: str, limit: int = 50) -> List[Dict]:
    """Load AF samples."""
    path = Path(path).expanduser()
    with open(path) as f:
        data = json.load(f)

    # Handle different formats
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
                if label in ["alignment_faking", "potential_faking"]:
                    processed.append({"text": text, "id": s.get("id", len(processed))})

    if len(processed) > limit:
        np.random.seed(42)
        indices = np.random.choice(len(processed), limit, replace=False)
        processed = [processed[i] for i in indices]

    return processed


def main():
    print("=" * 80)
    print("Vocabulary Masking Test")
    print("=" * 80)
    print("\nCRITICAL VALIDATION:")
    print("  Do AF-specific features survive keyword masking?")
    print("  Cluster probe FAILED this test (dropped to 0.5 AUROC)")
    print()

    config = {
        "base_model": "google/gemma-3-27b-it",
        "sae_repo": "google/gemma-scope-2-27b-it",
        "layer": 40,
        "af_path": "~/lightbright/data/gold_106.json",
        "n_samples": 50,
        "device": "cuda",
        "output_dir": "./vocab_masking_results"
    }

    # Load Phase 1 results to get AF-specific features
    print("1. Loading Phase 1 results...")
    with open("phase1_results/phase1_results.json") as f:
        phase1_results = json.load(f)

    layer_data = phase1_results["layers"]["layer_40"]

    # Get AF-specific feature IDs (top 100 for speed)
    af_specific_features = [
        f["feature_id"] for f in layer_data["top_features"]
        if f["differential"] > 0 and f["cohens_d"] > 0.5
    ][:100]  # Top 100 for this test

    print(f"   Testing top {len(af_specific_features)} AF-specific features")

    # Load data
    print("\n2. Loading AF samples...")
    af_samples = load_af_samples(config["af_path"], config["n_samples"])
    print(f"   Loaded {len(af_samples)} AF samples")

    # Mask keywords
    print("\n3. Masking RLHF keywords...")
    masked_samples = []
    total_keywords_masked = 0

    for sample in af_samples:
        masked_text, n_masked = mask_keywords(sample["text"], RLHF_KEYWORDS)
        total_keywords_masked += n_masked
        masked_samples.append({
            "original": sample["text"],
            "masked": masked_text,
            "n_keywords": n_masked,
            "id": sample["id"]
        })

    avg_masked = total_keywords_masked / len(masked_samples)
    print(f"   Masked {total_keywords_masked} keywords ({avg_masked:.1f} per sample)")

    # Load model
    print(f"\n4. Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # Load SAE
    print(f"   Loading SAE layer {config['layer']}...")
    sae = ResidPostSAE(config["sae_repo"], config["layer"])
    sae = sae.to(config["device"], dtype=torch.bfloat16)

    # Extract features from ORIGINAL and MASKED
    print("\n5. Extracting features...")

    original_features = []
    for sample in tqdm(masked_samples, desc="Original"):
        feats = extract_sae_features(
            model, sae, tokenizer, sample["original"],
            config["layer"], config["device"]
        )
        original_features.append(feats)

    masked_features = []
    for sample in tqdm(masked_samples, desc="Masked"):
        feats = extract_sae_features(
            model, sae, tokenizer, sample["masked"],
            config["layer"], config["device"]
        )
        masked_features.append(feats)

    original_features = np.stack(original_features)  # [n_samples, n_features]
    masked_features = np.stack(masked_features)

    # Compare activations for AF-specific features
    print("\n6. Analyzing feature robustness...")

    results = {
        "config": config,
        "n_samples": len(masked_samples),
        "n_features_tested": len(af_specific_features),
        "avg_keywords_masked": avg_masked,
        "features": []
    }

    for feat_idx in af_specific_features:
        original_acts = original_features[:, feat_idx]
        masked_acts = masked_features[:, feat_idx]

        # Mean activation
        orig_mean = original_acts.mean()
        mask_mean = masked_acts.mean()

        # Retention percentage
        retention = mask_mean / (orig_mean + 1e-8) if orig_mean > 0.1 else 0.0

        # Correlation
        if orig_mean > 0.1 and mask_mean > 0.1:
            correlation = np.corrcoef(original_acts, masked_acts)[0, 1]
        else:
            correlation = 0.0

        results["features"].append({
            "feature_id": int(feat_idx),
            "original_mean": float(orig_mean),
            "masked_mean": float(mask_mean),
            "retention": float(retention),
            "correlation": float(correlation),
            "robust": bool(retention > 0.7)  # >70% retention = robust
        })

    # Summary statistics
    retentions = [f["retention"] for f in results["features"]]
    robust_features = [f for f in results["features"] if f["robust"]]

    results["summary"] = {
        "mean_retention": float(np.mean(retentions)),
        "median_retention": float(np.median(retentions)),
        "n_robust_features": len(robust_features),
        "pct_robust": len(robust_features) / len(af_specific_features)
    }

    # Save results
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(output_dir / "vocab_masking_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nTested {len(af_specific_features)} top AF-specific features")
    print(f"Masked {avg_masked:.1f} keywords per sample")

    print(f"\nRetention statistics:")
    print(f"  Mean retention: {results['summary']['mean_retention']:.1%}")
    print(f"  Median retention: {results['summary']['median_retention']:.1%}")
    print(f"  Robust features (>70% retention): {results['summary']['n_robust_features']}/{len(af_specific_features)} ({results['summary']['pct_robust']:.1%})")

    print(f"\nTop 10 most robust features:")
    robust_sorted = sorted(results["features"], key=lambda x: x["retention"], reverse=True)
    for i, feat in enumerate(robust_sorted[:10], 1):
        print(f"  {i}. Feature {feat['feature_id']:5d}: "
              f"retention={feat['retention']:.1%}, "
              f"orig={feat['original_mean']:6.1f}, "
              f"masked={feat['masked_mean']:6.1f}")

    print(f"\nTop 10 least robust features (vocabulary-dependent):")
    for i, feat in enumerate(robust_sorted[-10:], 1):
        print(f"  {i}. Feature {feat['feature_id']:5d}: "
              f"retention={feat['retention']:.1%}, "
              f"orig={feat['original_mean']:6.1f}, "
              f"masked={feat['masked_mean']:6.1f}")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if results['summary']['pct_robust'] > 0.5:
        print(f"\n✅ SUCCESS: {results['summary']['pct_robust']:.0%} of features are robust!")
        print("   Features maintain activation when keywords masked")
        print("   They encode INTENT (reasoning structure), not vocabulary")
        print("\n   NEXT: Build interpretable detector from robust features")
    else:
        print(f"\n❌ FAILURE: Only {results['summary']['pct_robust']:.0%} of features are robust")
        print("   Features collapse when keywords masked")
        print("   They detect subtle vocabulary patterns, not intent")
        print("\n   CONCLUSION: Intent NOT in SAE activations")
        print("   NEXT: Try attention patterns or circuit analysis")

    return results


if __name__ == "__main__":
    results = main()
