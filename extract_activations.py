#!/usr/bin/env python3
"""
Extract SAE activations from Gemma-3-27B for alignment faking detection.

Usage:
    python extract_activations.py --input data.json --output activations/

Input JSON format:
    {
        "samples": [
            {"text": "reasoning trace...", "label": "potential_faking"},
            {"text": "another trace...", "label": "aligned"}
        ]
    }

Output:
    activations/
        activations.npy  - [n_samples, 16384] max activations
        labels.npy       - [n_samples] binary labels (1=AF, 0=aligned)
        metadata.json    - extraction metadata
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
CONFIG = {
    "model_name": "google/gemma-3-27b-it",
    "sae_repo": "google/gemma-scope-2-27b-it",
    "layer": 40,
    "n_features": 16384,
    "max_length": 2048,
}


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

        print(f"Loaded SAE layer {layer}: {self.w_enc.shape}")

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

    @property
    def num_features(self):
        return self.w_enc.shape[1]


def load_data(input_path):
    """Load input JSON data."""
    with open(input_path) as f:
        data = json.load(f)

    samples = data.get("samples", data)
    if isinstance(samples, dict):
        samples = list(samples.values())

    processed = []
    for idx, s in enumerate(samples):
        text = s.get("text", "") or s.get("scratchpad", "")
        label = s.get("label", "")

        # Explicit label validation
        if label == "potential_faking":
            is_af = 1
        elif label == "aligned":
            is_af = 0
        else:
            raise ValueError(
                f"Sample {idx}: Unknown label '{label}'. "
                f"Expected 'potential_faking' or 'aligned'"
            )

        processed.append({"text": text, "label": label, "is_af": is_af})

    n_af = sum(s["is_af"] for s in processed)
    print(f"Loaded {len(processed)} samples ({n_af} AF, {len(processed) - n_af} aligned)")
    return processed


def load_model():
    """Load Gemma model."""
    print(f"Loading {CONFIG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded ({torch.cuda.memory_allocated()/1024**3:.1f} GB VRAM)")
    return model, tokenizer


def extract_activations(model, tokenizer, sae, samples):
    """Extract max SAE activations for each sample."""
    n_samples = len(samples)
    n_features = sae.num_features
    layer = CONFIG["layer"]

    max_activations = np.zeros((n_samples, n_features), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int32)

    print(f"Extracting activations for {n_samples} samples...")

    for idx, sample in enumerate(tqdm(samples)):
        activations = {}

        def hook(module, input, output):
            if isinstance(output, tuple):
                activations["out"] = output[0].detach()
            else:
                activations["out"] = output.detach()

        # Hook into the target layer
        target = model.model.layers[layer]
        handle = target.register_forward_hook(hook)

        # Tokenize and forward pass
        inputs = tokenizer(
            sample["text"],
            return_tensors="pt",
            truncation=True,
            max_length=CONFIG["max_length"]
        ).to(model.device)

        with torch.no_grad():
            model(**inputs)

        handle.remove()

        # Encode with SAE and take max over sequence
        hidden = activations["out"][0]
        sae_acts = sae.encode(hidden.to(sae.w_enc.dtype))
        sae_acts_np = sae_acts.cpu().float().numpy()

        max_activations[idx] = sae_acts_np.max(axis=0)
        labels[idx] = sample["is_af"]

        del activations
        torch.cuda.empty_cache()

    return max_activations, labels


def main():
    parser = argparse.ArgumentParser(description="Extract SAE activations")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    samples = load_data(args.input)

    # Load model and SAE
    model, tokenizer = load_model()

    print(f"Loading SAE for layer {CONFIG['layer']}...")
    sae = ResidPostSAE(CONFIG["sae_repo"], CONFIG["layer"])
    sae.to(model.device, dtype=torch.bfloat16)

    # Extract activations
    activations, labels = extract_activations(model, tokenizer, sae, samples)

    # Save outputs
    np.save(output_dir / "activations.npy", activations)
    np.save(output_dir / "labels.npy", labels)

    metadata = {
        "n_samples": len(samples),
        "n_features": CONFIG["n_features"],
        "layer": CONFIG["layer"],
        "n_af": int(labels.sum()),
        "n_aligned": int(len(labels) - labels.sum()),
        "model": CONFIG["model_name"],
        "sae": CONFIG["sae_repo"],
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to {output_dir}/")
    print(f"  activations.npy: {activations.shape}")
    print(f"  labels.npy: {labels.shape}")
    print(f"  metadata.json")


if __name__ == "__main__":
    main()
