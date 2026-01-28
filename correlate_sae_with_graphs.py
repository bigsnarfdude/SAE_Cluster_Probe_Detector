#!/usr/bin/env python3
"""
Correlate SAE Feature Activations with ReasoningFlow Graph Components

Tests whether the 2,382 AF-specific SAE features correlate with:
1. Node types (planning, reasoning, reflection, etc.)
2. Edge patterns (refute vs support, verification, etc.)
3. Full subgraph patterns (AF reasoning paths vs HN paths)

This answers: Do features detect RLHF-awareness NODES or deceptive PATHS?
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')


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


@torch.no_grad()
def extract_sae_features_by_position(model, sae, tokenizer, text: str, start: int, end: int,
                                      layer_idx: int, device: str) -> np.ndarray:
    """Extract SAE features for a specific text span."""
    # Get token positions corresponding to text span
    inputs = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True).to(device)
    outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer_idx + 1]
    sae_features = sae.encode(hidden)

    # For simplicity, use max activation across the span
    # In production, would map character positions to token positions
    max_acts = sae_features.max(dim=1)[0]
    return max_acts.cpu().float().numpy()[0]


def load_annotated_graph(file_path: Path) -> Dict:
    """Load annotated ReasoningFlow graph."""
    with open(file_path) as f:
        return json.load(f)


def extract_node_type_counts(graph: Dict) -> Dict[str, int]:
    """Count nodes by type in the graph."""
    counts = {}
    for node in graph["nodes"]:
        if node.get("annotation", False):  # Only annotated nodes
            label = node["label"]
            counts[label] = counts.get(label, 0) + 1
    return counts


def extract_edge_type_counts(graph: Dict) -> Dict[str, int]:
    """Count edges by type in the graph."""
    counts = {}
    for edge in graph.get("edges", []):
        label = edge["label"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def has_subgraph_pattern(graph: Dict, pattern_name: str) -> bool:
    """Check if graph contains a specific reasoning pattern."""
    edges = graph.get("edges", [])
    nodes = {n["id"]: n for n in graph["nodes"]}

    if pattern_name == "verification":
        # Look for: planning node → frontier-verify edge → evaluation node
        for edge in edges:
            if "verify" in edge["label"].lower():
                return True
        return False

    elif pattern_name == "backtracking":
        # Look for: refute or correction edges
        for edge in edges:
            label = edge["label"].lower()
            if "refute" in label or "correction" in label:
                return True
        return False

    elif pattern_name == "conflict_pattern":
        # AF pattern: reasoning → refute → conclusion
        for edge in edges:
            if "refute" in edge["label"].lower():
                return True
        return False

    elif pattern_name == "support_pattern":
        # HN pattern: reasoning → support → conclusion
        for edge in edges:
            if "support" in edge["label"].lower():
                return True
        return False

    return False


def correlate_features_with_nodes(feature_activations: np.ndarray,
                                   node_counts: List[Dict[str, int]],
                                   node_type: str) -> Tuple[float, float]:
    """Correlate feature activations with node type presence."""
    node_presence = np.array([counts.get(node_type, 0) for counts in node_counts])

    if node_presence.sum() == 0:
        return 0.0, 1.0  # No correlation if node type never appears

    # Pearson correlation
    corr, pval = pearsonr(feature_activations, node_presence)
    return corr, pval


def main():
    print("=" * 80)
    print("Correlating SAE Features with ReasoningFlow Graph Components")
    print("=" * 80)
    print("\nHYPOTHESIS:")
    print("  If 91% FPR on hard negatives, features likely detect:")
    print("  ✓ Shared nodes (RLHF-awareness) → HIGH correlation")
    print("  ✗ Discriminative edges (refute vs support) → LOW correlation")
    print()

    config = {
        "base_model": "google/gemma-3-27b-it",
        "sae_repo": "google/gemma-scope-2-27b-it",
        "layer": 40,
        "phase1_results": "evidence_phase1_corrected.json",
        "annotation_dir": "./reasoningflow_annotation_samples",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir": "./reasoningflow_correlation_results"
    }

    # Check if annotations exist
    annotation_dir = Path(config["annotation_dir"])
    if not annotation_dir.exists():
        print(f"ERROR: Annotation directory not found: {annotation_dir}")
        print("\nPlease annotate samples first using:")
        print("  cd reasoningflow")
        print("  python annotator_webapp.py")
        return

    # Get annotated files
    annotated_files = list(annotation_dir.glob("*.json"))
    if len(annotated_files) == 0:
        print("ERROR: No annotated JSON files found")
        return

    # Filter out guide and README
    annotated_files = [f for f in annotated_files if f.stem.startswith(("af_", "hard_neg_"))]

    if len(annotated_files) == 0:
        print("ERROR: No AF or hard negative samples found")
        return

    print(f"1. Found {len(annotated_files)} annotated samples")

    # Load AF-specific feature IDs
    print("\n2. Loading AF-specific feature IDs...")
    with open(config["phase1_results"]) as f:
        results = json.load(f)

    layer_data = results["layers"]["layer_40"]
    af_feature_ids = []
    for feat in layer_data["top_features"]:
        if feat["differential"] > 0 and feat["cohens_d"] > 0.5:
            af_feature_ids.append(feat["feature_id"])

    print(f"   Using {len(af_feature_ids)} AF-specific features")

    # Load model
    print(f"\n3. Loading model and SAE...")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    sae = ResidPostSAE(config["sae_repo"], config["layer"])
    sae = sae.to(config["device"], dtype=torch.bfloat16)

    # Extract features and graph statistics for each sample
    print("\n4. Extracting SAE features and graph statistics...")

    samples_data = []

    for file_path in tqdm(annotated_files, desc="Processing"):
        graph = load_annotated_graph(file_path)
        text = graph["raw_text"]["response"]

        # Extract SAE features
        sae_features = extract_sae_features(model, sae, tokenizer, text, config["layer"], config["device"])

        # Extract graph statistics
        node_counts = extract_node_type_counts(graph)
        edge_counts = extract_edge_type_counts(graph)

        # Check for reasoning patterns
        has_verification = has_subgraph_pattern(graph, "verification")
        has_backtracking = has_subgraph_pattern(graph, "backtracking")
        has_conflict = has_subgraph_pattern(graph, "conflict_pattern")
        has_support = has_subgraph_pattern(graph, "support_pattern")

        samples_data.append({
            "file": file_path.name,
            "category": graph["metadata"]["category"],
            "is_af": graph["metadata"]["is_af"],
            "sae_features": sae_features[af_feature_ids],  # Only AF-specific features
            "node_counts": node_counts,
            "edge_counts": edge_counts,
            "has_verification": has_verification,
            "has_backtracking": has_backtracking,
            "has_conflict": has_conflict,
            "has_support": has_support
        })

    print(f"\n5. Analyzing correlations...")

    # Prepare data matrices
    feature_matrix = np.stack([s["sae_features"] for s in samples_data])  # (n_samples, n_features)

    # Node type analysis
    node_types = ["planning", "reasoning", "reflection", "fact", "restatement"]
    node_correlations = {}

    for node_type in node_types:
        node_presence = np.array([s["node_counts"].get(node_type, 0) for s in samples_data])

        if node_presence.sum() > 0:
            # Correlate each feature with node type presence
            correlations = []
            for feature_idx in range(feature_matrix.shape[1]):
                corr, pval = pearsonr(feature_matrix[:, feature_idx], node_presence)
                if pval < 0.05:  # Significant correlation
                    correlations.append((af_feature_ids[feature_idx], corr, pval))

            node_correlations[node_type] = {
                "n_samples_with_node": int(node_presence.sum()),
                "n_significant_features": len(correlations),
                "top_features": sorted(correlations, key=lambda x: abs(x[1]), reverse=True)[:5]
            }

    # Edge pattern analysis
    edge_patterns = ["refute", "support", "verify"]
    edge_correlations = {}

    for pattern in edge_patterns:
        pattern_presence = np.array([
            any(pattern in edge_label.lower() for edge_label in s["edge_counts"].keys())
            for s in samples_data
        ]).astype(int)

        if pattern_presence.sum() > 0:
            correlations = []
            for feature_idx in range(feature_matrix.shape[1]):
                corr, pval = pearsonr(feature_matrix[:, feature_idx], pattern_presence)
                if pval < 0.05:
                    correlations.append((af_feature_ids[feature_idx], corr, pval))

            edge_correlations[pattern] = {
                "n_samples_with_edge": int(pattern_presence.sum()),
                "n_significant_features": len(correlations),
                "top_features": sorted(correlations, key=lambda x: abs(x[1]), reverse=True)[:5]
            }

    # Subgraph pattern analysis
    conflict_presence = np.array([s["has_conflict"] for s in samples_data]).astype(int)
    support_presence = np.array([s["has_support"] for s in samples_data]).astype(int)

    # Save results
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    results = {
        "config": config,
        "n_samples": len(samples_data),
        "n_af_samples": sum(1 for s in samples_data if s["is_af"]),
        "n_hn_samples": sum(1 for s in samples_data if not s["is_af"]),
        "n_features_tested": len(af_feature_ids),
        "node_correlations": node_correlations,
        "edge_correlations": edge_correlations,
        "pattern_summary": {
            "conflict_pattern": {
                "n_samples": int(conflict_presence.sum()),
                "samples": [s["file"] for s in samples_data if s["has_conflict"]]
            },
            "support_pattern": {
                "n_samples": int(support_presence.sum()),
                "samples": [s["file"] for s in samples_data if s["has_support"]]
            }
        }
    }

    with open(output_dir / "correlation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nSamples analyzed: {len(samples_data)}")
    print(f"  AF samples: {results['n_af_samples']}")
    print(f"  Hard negatives: {results['n_hn_samples']}")

    print(f"\n--- Node Type Correlations ---")
    for node_type, data in node_correlations.items():
        print(f"\n{node_type.upper()}:")
        print(f"  Samples with this node: {data['n_samples_with_node']}/{len(samples_data)}")
        print(f"  Features significantly correlated: {data['n_significant_features']}/{len(af_feature_ids)}")
        if data['top_features']:
            print(f"  Top correlated features:")
            for feat_id, corr, pval in data['top_features']:
                print(f"    Feature {feat_id}: r={corr:.3f}, p={pval:.4f}")

    print(f"\n--- Edge Pattern Correlations ---")
    for pattern, data in edge_correlations.items():
        print(f"\n{pattern.upper()}:")
        print(f"  Samples with this edge: {data['n_samples_with_edge']}/{len(samples_data)}")
        print(f"  Features significantly correlated: {data['n_significant_features']}/{len(af_feature_ids)}")
        if data['top_features']:
            print(f"  Top correlated features:")
            for feat_id, corr, pval in data['top_features']:
                print(f"    Feature {feat_id}: r={corr:.3f}, p={pval:.4f}")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    # Calculate relative correlation strength
    total_node_correlations = sum(d['n_significant_features'] for d in node_correlations.values())
    total_edge_correlations = sum(d['n_significant_features'] for d in edge_correlations.values())

    print(f"\nTotal feature-node correlations: {total_node_correlations}")
    print(f"Total feature-edge correlations: {total_edge_correlations}")

    if total_node_correlations > 2 * total_edge_correlations:
        print("\n✓ HYPOTHESIS CONFIRMED:")
        print("  Features correlate MORE with node types (RLHF-awareness)")
        print("  Features correlate LESS with edge patterns (conflict vs support)")
        print("\n  CONCLUSION:")
        print("  Features detect 'RLHF-reasoning' nodes, not deceptive intent paths")
        print("  This explains 91% FPR on hard negatives (shared nodes)")
    else:
        print("\n✗ HYPOTHESIS REJECTED:")
        print("  Features correlate more with edge patterns than nodes")
        print("  Further investigation needed")

    print(f"\nResults saved to: {output_dir}/correlation_results.json")


if __name__ == "__main__":
    main()
