#!/usr/bin/env python3
"""
Phase 1: Feature Discovery

Quick experiment to identify SAE features that discriminate AF from hard negatives.
This is the critical first step - if this fails, intent is not in SAE features.

Runtime: ~1 hour on A100 GPU
"""

import json
import os
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Config:
    # Models
    base_model: str = "google/gemma-3-27b-it"
    sae_repo: str = "google/gemma-scope-2-27b-it"

    # Layers to analyze
    layers: List[int] = None

    # Data paths (adjust for nigel)
    af_samples_path: str = "~/lightbright/data/gold_106.json"
    hard_neg_path: str = "~/af-detection-benchmark/data/classes/compliant_hard_negative.json"

    # Sampling
    n_af_samples: int = 50
    n_hard_neg_samples: int = 100

    # Analysis
    top_k_features: int = 100  # Report top K discriminative features

    # Output
    output_dir: str = "./phase1_results"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 2048

    def __post_init__(self):
        if self.layers is None:
            self.layers = [40]  # Start with single layer for speed


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
        self.w_dec = weights["w_dec"]
        self.b_dec = weights["b_dec"]
        self.threshold = weights.get("threshold", None)

        print(f"  Loaded SAE layer {layer}: {self.w_enc.shape}")

    def to(self, device, dtype=None):
        self.w_enc = self.w_enc.to(device, dtype=dtype)
        self.b_enc = self.b_enc.to(device, dtype=dtype)
        self.w_dec = self.w_dec.to(device, dtype=dtype)
        self.b_dec = self.b_dec.to(device, dtype=dtype)
        if self.threshold is not None:
            self.threshold = self.threshold.to(device, dtype=dtype)
        return self

    def encode(self, x):
        """Encode residual stream to sparse features."""
        pre_acts = x @ self.w_enc + self.b_enc
        if self.threshold is not None:
            return F.relu(pre_acts - self.threshold)
        return F.relu(pre_acts)

    @property
    def num_features(self):
        return self.w_enc.shape[1]


def load_dataset(path: str, label_filter: str = None, limit: int = None) -> List[Dict]:
    """Load and filter dataset."""
    path = Path(path).expanduser()

    if not path.exists():
        print(f"WARNING: {path} not found, using placeholder")
        return []

    print(f"Loading {path}")

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
        text = s.get("text", "") or s.get("scratchpad", "") or s.get("content", "")
        if not text:
            continue

        label = s.get("label", "")
        is_af = 1 if label in ["alignment_faking", "potential_faking"] else 0

        if 'is_af' in s:
            is_af = s['is_af']

        # Filter by label
        if label_filter:
            if (label_filter == "af" and not is_af) or (label_filter == "aligned" and is_af):
                continue

        processed.append({
            "text": text,
            "is_af": is_af,
            "label": label,
            "id": s.get("id", len(processed))
        })

    if limit and len(processed) > limit:
        # Random sample
        np.random.seed(42)
        indices = np.random.choice(len(processed), limit, replace=False)
        processed = [processed[i] for i in indices]

    print(f"  Loaded {len(processed)} samples")
    return processed


@torch.no_grad()
def extract_sae_features(
    model,
    sae,
    tokenizer,
    text: str,
    layer_idx: int,
    device: str,
    max_length: int = 2048
) -> np.ndarray:
    """
    Extract SAE features from model's residual stream at specific layer.

    Returns:
        max_activations: [n_features] - max activation across sequence
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True
    ).to(device)

    # Get hidden states
    outputs = model(**inputs, output_hidden_states=True)

    # Extract residual stream at target layer
    # outputs.hidden_states is tuple of (input_embeds, layer1, layer2, ..., layerN)
    # layer_idx 0 = input embeddings, layer_idx N = output of layer N
    hidden = outputs.hidden_states[layer_idx + 1]  # +1 because index 0 is input embeds

    # Encode with SAE
    sae_features = sae.encode(hidden)  # [batch, seq_len, n_features]

    # Max pool over sequence
    max_acts = sae_features.max(dim=1)[0]  # [batch, n_features]

    return max_acts.cpu().float().numpy()[0]  # Convert to float32 for numpy


def analyze_features(
    af_features: np.ndarray,
    hard_neg_features: np.ndarray,
    layer: int,
    config: Config
) -> Dict:
    """
    Analyze feature activation patterns to find discriminative features.

    Args:
        af_features: [n_af_samples, n_features]
        hard_neg_features: [n_hard_neg_samples, n_features]

    Returns:
        analysis results dict
    """
    print(f"\n  Analyzing layer {layer} features...")

    # Compute statistics
    af_mean = af_features.mean(axis=0)
    af_std = af_features.std(axis=0)

    hard_neg_mean = hard_neg_features.mean(axis=0)
    hard_neg_std = hard_neg_features.std(axis=0)

    # Differential activation (key metric)
    diff = af_mean - hard_neg_mean

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((af_std**2 + hard_neg_std**2) / 2)
    cohens_d = diff / (pooled_std + 1e-8)

    # Find top discriminative features
    top_indices = np.argsort(np.abs(diff))[::-1][:config.top_k_features]

    # Statistics for top features
    top_features = []
    for idx in top_indices[:20]:  # Detailed stats for top 20
        top_features.append({
            "feature_id": int(idx),
            "af_mean": float(af_mean[idx]),
            "af_std": float(af_std[idx]),
            "hard_neg_mean": float(hard_neg_mean[idx]),
            "hard_neg_std": float(hard_neg_std[idx]),
            "differential": float(diff[idx]),
            "cohens_d": float(cohens_d[idx]),
            "activation_ratio": float(af_mean[idx] / (hard_neg_mean[idx] + 1e-8))
        })

    # Count significant features BY DIRECTION
    # Positive Cohen's d = fires MORE on AF (what we want for intent detection)
    # Negative Cohen's d = fires MORE on hard negatives (vocabulary features)
    n_af_specific = (cohens_d > 0.5).sum()  # Intent features
    n_hn_specific = (cohens_d < -0.5).sum()  # Vocabulary features
    n_total_discriminative = (np.abs(cohens_d) > 0.5).sum()

    results = {
        "layer": layer,
        "n_features": af_features.shape[1],
        "n_af_samples": af_features.shape[0],
        "n_hard_neg_samples": hard_neg_features.shape[0],
        "top_features": top_features,
        "n_af_specific_features": int(n_af_specific),  # INTENT features
        "n_hn_specific_features": int(n_hn_specific),  # VOCABULARY features
        "n_total_discriminative": int(n_total_discriminative),
        "max_differential": float(np.abs(diff).max()),
        "max_cohens_d": float(np.abs(cohens_d).max()),
        "summary": {
            "mean_af_activation": float(af_mean.mean()),
            "mean_hard_neg_activation": float(hard_neg_mean.mean()),
            "mean_differential": float(diff.mean()),
            "std_differential": float(diff.std())
        }
    }

    return results


def run_phase1(config: Config):
    """Main Phase 1 experiment."""

    print("=" * 80)
    print("Phase 1: Feature Discovery")
    print("=" * 80)
    print(f"\nGoal: Find SAE features that discriminate AF from hard negatives")
    print(f"Layers: {config.layers}")
    print(f"Samples: {config.n_af_samples} AF, {config.n_hard_neg_samples} hard negatives")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load datasets
    print("\n1. Loading datasets...")
    af_samples = load_dataset(
        config.af_samples_path,
        label_filter="af",
        limit=config.n_af_samples
    )
    hard_neg_samples = load_dataset(
        config.hard_neg_path,
        label_filter="aligned",
        limit=config.n_hard_neg_samples
    )

    if not af_samples or not hard_neg_samples:
        print("ERROR: Could not load datasets. Check paths.")
        return

    # Load model
    print(f"\n2. Loading base model: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # Results container
    all_results = {
        "config": {
            "base_model": config.base_model,
            "sae_repo": config.sae_repo,
            "layers": config.layers,
            "n_af_samples": len(af_samples),
            "n_hard_neg_samples": len(hard_neg_samples),
        },
        "layers": {}
    }

    # Process each layer
    for layer in config.layers:
        print(f"\n3. Processing layer {layer}...")

        # Load SAE
        print(f"  Loading SAE...")
        sae = ResidPostSAE(config.sae_repo, layer)
        sae = sae.to(config.device, dtype=torch.bfloat16)

        # Extract features for AF samples
        print(f"  Extracting AF features...")
        af_features = []
        for sample in tqdm(af_samples, desc="AF samples"):
            feats = extract_sae_features(
                model, sae, tokenizer, sample['text'],
                layer, config.device, config.max_length
            )
            af_features.append(feats)

        af_features = np.stack(af_features)

        # Extract features for hard negatives
        print(f"  Extracting hard negative features...")
        hard_neg_features = []
        for sample in tqdm(hard_neg_samples, desc="Hard negatives"):
            feats = extract_sae_features(
                model, sae, tokenizer, sample['text'],
                layer, config.device, config.max_length
            )
            hard_neg_features.append(feats)

        hard_neg_features = np.stack(hard_neg_features)

        # Analyze
        layer_results = analyze_features(af_features, hard_neg_features, layer, config)
        all_results["layers"][f"layer_{layer}"] = layer_results

        # Print summary
        print(f"\n  Layer {layer} Summary:")
        print(f"    AF-specific features (d > 0.5): {layer_results['n_af_specific_features']}")
        print(f"    HN-specific features (d < -0.5): {layer_results['n_hn_specific_features']}")
        print(f"    Total discriminative: {layer_results['n_total_discriminative']}")
        print(f"    Max differential: {layer_results['max_differential']:.2f}")
        print(f"    Max Cohen's d: {layer_results['max_cohens_d']:.2f}")
        print(f"\n    Top 5 AF-specific features:")
        # Filter for positive differential (fires more on AF)
        af_specific_feats = [f for f in layer_results['top_features'] if f['differential'] > 0]
        for feat in af_specific_feats[:5]:
            print(f"      Feature {feat['feature_id']:5d}: "
                  f"diff=+{feat['differential']:6.2f}, "
                  f"AF={feat['af_mean']:6.2f}, "
                  f"HN={feat['hard_neg_mean']:6.2f}, "
                  f"d={feat['cohens_d']:6.2f}")

    # Save results
    output_file = output_dir / "phase1_results.json"
    print(f"\n4. Saving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Final summary
    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE")
    print("=" * 80)

    for layer in config.layers:
        layer_data = all_results["layers"][f"layer_{layer}"]
        print(f"\nLayer {layer}:")
        print(f"  AF-specific features: {layer_data['n_af_specific_features']}")
        print(f"  HN-specific features: {layer_data['n_hn_specific_features']}")
        print(f"  Total discriminative: {layer_data['n_total_discriminative']}")

    # Verdict - use AF-specific features (not total discriminative)
    max_af_specific = max(
        all_results["layers"][f"layer_{layer}"]["n_af_specific_features"]
        for layer in config.layers
    )

    print("\n" + "=" * 80)
    if max_af_specific >= 10:
        print("✅ SUCCESS: Found AF-specific intent features!")
        print(f"   {max_af_specific} features fire MORE on AF than hard negatives")
        print(f"   (These encode intent, not vocabulary)")
        print("\nNEXT STEPS:")
        print("1. Inspect top AF-specific features (GemmaScope dashboards)")
        print("2. Test vocabulary masking robustness")
        print("3. Implement Phase 2: Causal validation (activation patching)")
    else:
        print("❌ FAILURE: No AF-specific intent features found")
        print(f"   Only {max_af_specific} features fire more on AF than hard negatives")
        print("\nCONCLUSION:")
        print("Intent signal is NOT in single-layer SAE features")
        print("Features that discriminate are vocabulary detectors (fire on hard negatives)")
        print("\nNEXT EXPERIMENTS:")
        print("- Experiment 1.2: Attention pattern analysis")
        print("- Experiment 1.3: Multi-layer trajectory analysis")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[40])
    parser.add_argument("--n_af", type=int, default=50)
    parser.add_argument("--n_hard_neg", type=int, default=100)
    parser.add_argument("--output_dir", default="./phase1_results")

    args = parser.parse_args()

    config = Config(
        layers=args.layers,
        n_af_samples=args.n_af,
        n_hard_neg_samples=args.n_hard_neg,
        output_dir=args.output_dir
    )

    results = run_phase1(config)
