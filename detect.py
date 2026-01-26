#!/usr/bin/env python3
"""
Run alignment faking detection on new samples.

Usage:
    # From pre-extracted activations
    python detect.py --activations activations.npy --model checkpoints/

    # From raw JSON (requires GPU)
    python detect.py --input samples.json --model checkpoints/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cluster import FeatureClusterer
from model import ClusterAttentionNet


def load_model(model_dir, device):
    """Load trained model and clusterer."""
    model_dir = Path(model_dir)

    # Load clusterer
    clusterer = FeatureClusterer()
    clusterer.load(model_dir / "clusterer.npz")

    # Load model
    model = ClusterAttentionNet(
        n_clusters=clusterer.n_clusters,
        embed_dim=32,
        n_heads=4,
        dropout=0.3
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.eval()

    return model, clusterer


def detect_from_activations(activations, model, clusterer, device):
    """
    Run detection on pre-extracted activations.

    Args:
        activations: [n_samples, 16384] SAE activations
        model: Trained ClusterAttentionNet
        clusterer: Fitted FeatureClusterer

    Returns:
        probabilities: [n_samples] P(alignment_faking)
        predictions: [n_samples] binary predictions
    """
    # Transform to cluster representation
    X = clusterer.transform(activations)
    X_tensor = torch.FloatTensor(X).to(device)

    # Get predictions
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits.squeeze()).cpu().numpy()

    predictions = (probs > 0.5).astype(int)
    return probs, predictions


def main():
    parser = argparse.ArgumentParser(description="Detect alignment faking")
    parser.add_argument("--activations", "-a", help="Pre-extracted activations (.npy)")
    parser.add_argument("--input", "-i", help="Raw input JSON (requires GPU + extraction)")
    parser.add_argument("--model", "-m", required=True, help="Model directory")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model}...")
    model, clusterer = load_model(args.model, device)

    # Get activations
    if args.activations:
        print(f"Loading activations from {args.activations}...")
        activations = np.load(args.activations)
    elif args.input:
        print(f"Extracting activations from {args.input}...")
        print("This requires a GPU with ~24GB VRAM")

        # Import extraction module
        from extract_activations import load_data, load_model as load_gemma, ResidPostSAE, CONFIG

        samples = load_data(args.input)
        gemma, tokenizer = load_gemma()

        sae = ResidPostSAE(CONFIG["sae_repo"], CONFIG["layer"])
        sae.to(gemma.device, dtype=torch.bfloat16)

        from extract_activations import extract_activations
        activations, _ = extract_activations(gemma, tokenizer, sae, samples)
    else:
        parser.error("Must provide either --activations or --input")

    # Run detection
    print(f"\nRunning detection on {len(activations)} samples...")
    probs, preds = detect_from_activations(activations, model, clusterer, device)

    # Validate results
    if len(preds) == 0:
        print("\nNo samples to analyze")
        return

    # Print results
    n_af = preds.sum()
    print(f"\n{'='*50}")
    print(f"DETECTION RESULTS")
    print(f"{'='*50}")
    print(f"Samples analyzed: {len(preds)}")
    print(f"Alignment faking detected: {n_af} ({100*n_af/len(preds):.1f}%)")
    print(f"Aligned: {len(preds) - n_af} ({100*(len(preds)-n_af)/len(preds):.1f}%)")

    # Show high-confidence detections
    high_conf = np.where(probs > 0.8)[0]
    if len(high_conf) > 0:
        print(f"\nHigh confidence AF detections (p > 0.8): {len(high_conf)}")
        for idx in high_conf[:10]:  # Show first 10
            print(f"  Sample {idx}: p={probs[idx]:.3f}")

    # Save output
    if args.output:
        results = {
            "n_samples": len(preds),
            "n_af_detected": int(n_af),
            "threshold": args.threshold,
            "samples": [
                {"index": i, "probability": float(p), "prediction": int(pred)}
                for i, (p, pred) in enumerate(zip(probs, preds))
            ]
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
