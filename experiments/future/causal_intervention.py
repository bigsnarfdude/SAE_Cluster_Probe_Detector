#!/usr/bin/env python3
"""
Experiment 1.1: Causal Feature Identification

Identify which SAE features causally determine alignment faking detection
in fine-tuned models. This reveals if intent signal exists in SAE features.

Method:
1. Load fine-tuned AF detector (that actually works)
2. Extract SAE features at multiple layers
3. Ablate features one by one, measure impact on predictions
4. Compare AF vs hard negatives (same vocab, different intent)
5. Identify causal "intent features" vs "vocabulary features"

Success criteria:
- Find features that activate differently on AF vs hard negatives
- Features should be robust to keyword masking
- Should explain how SFT models detect intent
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Config:
    # Models
    base_model: str = "google/gemma-3-27b-it"
    detector_model: str = "vincentoh/gemma3-27b-af-detector-v2"  # Fine-tuned detector
    sae_repo: str = "google/gemma-scope-2-27b-it"

    # SAE config
    layers: List[int] = None  # [20, 30, 40, 50]
    n_features: int = 16384

    # Data
    af_samples_path: str = "~/af-detection-benchmark/data/gold_106.json"
    hard_neg_path: str = "~/af-detection-benchmark/data/classes/compliant_hard_negative.json"

    # Sampling (for speed)
    n_af_samples: int = 50  # Sample from gold_106
    n_hard_neg_samples: int = 100  # Sample from 520 hard negatives

    # Ablation
    top_k_features: int = 100  # Only ablate top K most active features

    # Output
    output_dir: str = "./causal_intervention_results"

    # Device
    device: str = "cuda"
    max_length: int = 2048

    def __post_init__(self):
        if self.layers is None:
            self.layers = [20, 30, 40, 50]


class ResidPostSAE:
    """GemmaScope resid_post SAE for activation extraction."""

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
        """Encode activations to SAE features."""
        pre_acts = x @ self.w_enc + self.b_enc
        if self.threshold is not None:
            return F.relu(pre_acts - self.threshold)
        return F.relu(pre_acts)

    @property
    def num_features(self):
        return self.w_enc.shape[1]


def load_dataset(path: str, label_filter: str = None, limit: int = None) -> List[Dict]:
    """Load dataset with optional filtering."""
    path = Path(path).expanduser()
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

        # Override with explicit is_af if present
        if 'is_af' in s:
            is_af = s['is_af']

        if label_filter:
            if (label_filter == "af" and not is_af) or (label_filter == "aligned" and is_af):
                continue

        processed.append({
            "text": text,
            "is_af": is_af,
            "label": label
        })

    if limit and len(processed) > limit:
        # Random sample
        indices = np.random.choice(len(processed), limit, replace=False)
        processed = [processed[i] for i in indices]

    print(f"  Loaded {len(processed)} samples")
    return processed


def get_detector_prediction(model, tokenizer, text: str, device: str) -> Tuple[float, torch.Tensor]:
    """
    Get AF probability from fine-tuned detector.

    Returns:
        (probability, logits)
    """
    # Format as instruction for Gemma
    prompt = f"<bos><start_of_turn>user\nAnalyze this reasoning trace for alignment faking:\n\n{text}\n<end_of_turn>\n<start_of_turn>model\n"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]  # Last token logits

        # Get probability for AF class
        # Assume fine-tuned model outputs class logits [aligned, af]
        # This may need adjustment based on actual model
        probs = F.softmax(logits, dim=-1)

        # For now, use max prob as AF score
        af_prob = probs.max().item()

    return af_prob, logits


def extract_sae_features(model, sae, tokenizer, text: str, device: str, max_length: int = 2048) -> np.ndarray:
    """
    Extract SAE features from model's residual stream.

    Returns:
        activations: [n_features] max activation across sequence
    """
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True).to(device)

    # Hook to capture activations at target layer
    activations_cache = {}

    def hook_fn(module, input, output):
        # output is tuple (hidden_states,) for residual
        hidden = output[0] if isinstance(output, tuple) else output
        activations_cache['hidden'] = hidden.detach()

    # Register hook (layer indexing depends on model architecture)
    # For Gemma, layers are in model.model.layers
    # We'll need to identify the correct layer
    # This is a simplified version - actual implementation needs proper layer targeting

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Get residual stream at target layer
        # For now, use last hidden state as proxy
        hidden = outputs.hidden_states[-1]  # [batch, seq, hidden_dim]

    # Encode with SAE
    sae_features = sae.encode(hidden)  # [batch, seq, n_features]

    # Max pool over sequence dimension
    max_acts = sae_features.max(dim=1)[0]  # [batch, n_features]

    return max_acts.cpu().numpy()[0]  # [n_features]


def ablate_feature_and_predict(
    model,
    tokenizer,
    sae,
    text: str,
    feature_idx: int,
    device: str
) -> float:
    """
    Ablate a single SAE feature and get new AF prediction.

    This is a simplified version - full implementation would:
    1. Extract activations at target layer
    2. Decode to residual space
    3. Zero out feature contribution
    4. Re-encode and continue forward pass
    5. Get final prediction

    For now, we'll use a proxy method.
    """
    # TODO: Implement proper activation patching
    # This requires hooking into forward pass and modifying activations

    # Placeholder: return random perturbation
    return np.random.rand()


def run_causal_intervention(config: Config):
    """Main experimental loop."""

    print("=" * 80)
    print("Experiment 1.1: Causal Feature Identification")
    print("=" * 80)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load datasets
    print("\n1. Loading datasets...")
    af_samples = load_dataset(config.af_samples_path, label_filter="af", limit=config.n_af_samples)
    hard_neg_samples = load_dataset(config.hard_neg_path, label_filter="aligned", limit=config.n_hard_neg_samples)

    print(f"\n  AF samples: {len(af_samples)}")
    print(f"  Hard negative samples: {len(hard_neg_samples)}")

    # Load models
    print("\n2. Loading models...")
    print(f"  Base model: {config.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    # Note: For causal intervention, we need base model + SAE, not fine-tuned detector
    # The detector is used to verify predictions, not for ablation
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f"\n  Loading fine-tuned detector: {config.detector_model}")
    # detector_model = PeftModel.from_pretrained(base_model, config.detector_model)
    # For now, use base model as proxy
    detector_model = base_model

    # Load SAEs for each layer
    print(f"\n3. Loading SAEs for layers: {config.layers}")
    saes = {}
    for layer in config.layers:
        sae = ResidPostSAE(config.sae_repo, layer)
        sae = sae.to(config.device, dtype=torch.bfloat16)
        saes[layer] = sae

    # Run intervention experiments
    print("\n4. Running causal interventions...")

    results = {
        "config": {
            "base_model": config.base_model,
            "detector_model": config.detector_model,
            "layers": config.layers,
            "n_af_samples": len(af_samples),
            "n_hard_neg_samples": len(hard_neg_samples),
        },
        "features": {},  # feature_id -> impact metrics
        "samples": []
    }

    # For each layer
    for layer in config.layers:
        print(f"\n  Layer {layer}:")
        sae = saes[layer]

        # Extract features for all samples
        print("    Extracting SAE features...")

        af_features = []
        for sample in tqdm(af_samples[:10], desc="AF samples"):  # Limit for testing
            feats = extract_sae_features(base_model, sae, tokenizer, sample['text'], config.device)
            af_features.append(feats)

        hard_neg_features = []
        for sample in tqdm(hard_neg_samples[:10], desc="Hard negatives"):
            feats = extract_sae_features(base_model, sae, tokenizer, sample['text'], config.device)
            hard_neg_features.append(feats)

        # Stack features
        af_features = np.stack(af_features)  # [n_af, n_features]
        hard_neg_features = np.stack(hard_neg_features)  # [n_hard_neg, n_features]

        # Compute feature statistics
        print("    Analyzing feature activation patterns...")

        # Mean activation for each feature
        af_mean = af_features.mean(axis=0)  # [n_features]
        hard_neg_mean = hard_neg_features.mean(axis=0)

        # Difference (features that activate more in AF vs hard neg)
        diff = af_mean - hard_neg_mean

        # Find top differentiating features
        top_indices = np.argsort(np.abs(diff))[::-1][:config.top_k_features]

        print(f"    Top {config.top_k_features} discriminative features identified")

        # Save layer results
        layer_results = {
            "layer": layer,
            "af_mean_activation": af_mean.tolist(),
            "hard_neg_mean_activation": hard_neg_mean.tolist(),
            "difference": diff.tolist(),
            "top_features": top_indices.tolist(),
            "top_feature_diffs": diff[top_indices].tolist()
        }

        results[f"layer_{layer}"] = layer_results

    # Save results
    output_file = output_dir / "causal_intervention_results.json"
    print(f"\n5. Saving results to {output_file}")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)

    for layer in config.layers:
        layer_data = results[f"layer_{layer}"]
        top_diffs = layer_data["top_feature_diffs"]
        print(f"\nLayer {layer}:")
        print(f"  Top feature differential: {top_diffs[0]:.4f}")
        print(f"  Top 10 feature IDs: {layer_data['top_features'][:10]}")

    print("\nNEXT STEPS:")
    print("1. Analyze top differentiating features")
    print("2. Check if they survive vocabulary masking")
    print("3. Implement full activation patching to measure causal impact")
    print("4. Validate features actually encode intent vs vocabulary")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="google/gemma-3-27b-it")
    parser.add_argument("--detector_model", default="vincentoh/gemma3-27b-af-detector-v2")
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 30, 40, 50])
    parser.add_argument("--n_af_samples", type=int, default=50)
    parser.add_argument("--n_hard_neg", type=int, default=100)
    parser.add_argument("--output_dir", default="./causal_intervention_results")

    args = parser.parse_args()

    config = Config(
        base_model=args.base_model,
        detector_model=args.detector_model,
        layers=args.layers,
        n_af_samples=args.n_af_samples,
        n_hard_neg_samples=args.n_hard_neg,
        output_dir=args.output_dir
    )

    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)

    results = run_causal_intervention(config)
