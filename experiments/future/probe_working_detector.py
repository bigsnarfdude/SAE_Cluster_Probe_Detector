#!/usr/bin/env python3
"""
Probe the Working Detector

Examine what the fine-tuned AF detector actually learned.
By comparing AF vs hard negative samples (same keywords, different intent),
we can see WHERE the intent signal lives.

This tells us:
- Which layers encode intent
- Whether it's in activations, attention, or both
- Which tokens the model focuses on
- What type of signal to look for in SAEs

Runtime: ~10 minutes
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Config:
    # Models
    base_model: str = "google/gemma-3-27b-it"
    detector_model: str = "vincentoh/gemma3-27b-af-detector-v2"

    # Alternative: Use local trained detector on nigel
    use_local_detector: bool = True
    local_detector_path: str = "~/mindreader/models/gemma-27b-af-detector-with707"

    # Data
    af_samples_path: str = "~/lightbright/data/gold_106.json"
    hard_neg_path: str = "~/miscJan2026/af-detection-benchmark/data/classes/compliant_hard_negative.json"

    # Sampling (pick representative samples)
    n_pairs: int = 10  # Number of AF-HardNeg pairs to compare

    # Layers to analyze
    layers_to_probe: List[int] = None  # Will analyze all layers

    # Output
    output_dir: str = "./detector_probe_results"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 2048

    def __post_init__(self):
        if self.layers_to_probe is None:
            # Gemma-3-27B has 42 layers, focus on key layers
            self.layers_to_probe = [10, 20, 30, 40, 41]  # Early, mid, late


def load_dataset(path: str, limit: int = None) -> List[Dict]:
    """Load dataset."""
    path = Path(path).expanduser()

    if not path.exists():
        print(f"WARNING: {path} not found")
        return []

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

        processed.append({
            "text": text,
            "is_af": is_af,
            "label": label,
            "id": s.get("id", len(processed))
        })

    if limit and len(processed) > limit:
        np.random.seed(42)
        indices = np.random.choice(len(processed), limit, replace=False)
        processed = [processed[i] for i in indices]

    return processed


@torch.no_grad()
def extract_internals(
    model,
    tokenizer,
    text: str,
    layers_to_probe: List[int],
    device: str,
    max_length: int = 2048
) -> Dict:
    """
    Extract internal representations from model.

    Returns:
        {
            'tokens': token_ids,
            'token_strs': decoded tokens,
            'activations': {layer_idx: [seq_len, hidden_dim]},
            'attention': {layer_idx: [n_heads, seq_len, seq_len]},
            'logits': [vocab_size]
        }
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True
    ).to(device)

    # Get outputs with all internals
    outputs = model(
        **inputs,
        output_hidden_states=True,
        output_attentions=True
    )

    # Extract token strings
    token_ids = inputs['input_ids'][0].cpu().tolist()
    token_strs = [tokenizer.decode([tid]) for tid in token_ids]

    # Extract activations at key layers
    activations = {}
    for layer_idx in layers_to_probe:
        if layer_idx < len(outputs.hidden_states):
            # hidden_states[0] = embeddings, hidden_states[i+1] = layer i output
            hidden = outputs.hidden_states[layer_idx + 1]  # [batch, seq, hidden]
            activations[layer_idx] = hidden[0].cpu().float().numpy()  # [seq, hidden]

    # Extract attention patterns at key layers
    attention = {}
    for layer_idx in layers_to_probe:
        if layer_idx < len(outputs.attentions):
            attn = outputs.attentions[layer_idx]  # [batch, n_heads, seq, seq]
            attention[layer_idx] = attn[0].cpu().float().numpy()  # [n_heads, seq, seq]

    # Final logits
    logits = outputs.logits[0, -1, :].cpu().float().numpy()  # [vocab_size]

    return {
        'tokens': token_ids,
        'token_strs': token_strs,
        'activations': activations,
        'attention': attention,
        'logits': logits,
        'seq_len': len(token_ids)
    }


def compare_internals(af_internals: Dict, hn_internals: Dict) -> Dict:
    """
    Compare internal representations of AF vs hard negative.

    Find what's DIFFERENT when intent is present.
    """
    results = {
        'layers': {},
        'summary': {}
    }

    # Compare each layer
    for layer_idx in af_internals['activations'].keys():
        af_acts = af_internals['activations'][layer_idx]  # [seq_af, hidden]
        hn_acts = hn_internals['activations'][layer_idx]  # [seq_hn, hidden]

        # Mean activation difference (average over sequence)
        af_mean = af_acts.mean(axis=0)  # [hidden]
        hn_mean = hn_acts.mean(axis=0)  # [hidden]

        diff = np.abs(af_mean - hn_mean)

        # Find most discriminative neurons
        top_neurons = np.argsort(diff)[::-1][:100]

        # Compute layer-level statistics
        layer_results = {
            'mean_diff': float(diff.mean()),
            'max_diff': float(diff.max()),
            'top_neurons': top_neurons.tolist(),
            'top_neuron_diffs': diff[top_neurons].tolist(),
            'af_mean_activation': float(af_mean.mean()),
            'hn_mean_activation': float(hn_mean.mean())
        }

        # Compare attention patterns
        if layer_idx in af_internals['attention']:
            af_attn = af_internals['attention'][layer_idx]  # [n_heads, seq, seq]
            hn_attn = hn_internals['attention'][layer_idx]

            # Average attention per head
            af_attn_mean = af_attn.mean(axis=(1, 2))  # [n_heads]
            hn_attn_mean = hn_attn.mean(axis=(1, 2))

            attn_diff = np.abs(af_attn_mean - hn_attn_mean)

            layer_results['attention'] = {
                'mean_diff': float(attn_diff.mean()),
                'max_diff': float(attn_diff.max()),
                'top_heads': np.argsort(attn_diff)[::-1][:4].tolist(),
                'top_head_diffs': attn_diff[np.argsort(attn_diff)[::-1][:4]].tolist()
            }

        results['layers'][layer_idx] = layer_results

    # Summary: Which layer has strongest signal?
    layer_diffs = [(idx, results['layers'][idx]['max_diff'])
                   for idx in results['layers'].keys()]
    layer_diffs.sort(key=lambda x: x[1], reverse=True)

    results['summary'] = {
        'most_discriminative_layer': layer_diffs[0][0] if layer_diffs else None,
        'max_neuron_diff': layer_diffs[0][1] if layer_diffs else 0.0,
        'layer_ranking': layer_diffs
    }

    return results


def analyze_attention_focus(
    internals: Dict,
    label: str,
    output_dir: Path
) -> Dict:
    """
    Analyze which tokens get attention.

    For AF samples: Does it focus on self-model tokens? RLHF mentions?
    For hard negatives: Does it focus on user request tokens?
    """
    results = {
        'label': label,
        'seq_len': internals['seq_len'],
        'tokens': internals['token_strs'],
        'layers': {}
    }

    for layer_idx, attn in internals['attention'].items():
        # attn: [n_heads, seq, seq]
        # Average attention received by each token (column sum)
        attention_received = attn.mean(axis=0).sum(axis=0)  # [seq]

        # Top attended tokens
        top_token_indices = np.argsort(attention_received)[::-1][:10]

        top_tokens = []
        for idx in top_token_indices:
            if idx < len(internals['token_strs']):
                top_tokens.append({
                    'position': int(idx),
                    'token': internals['token_strs'][idx],
                    'attention_score': float(attention_received[idx])
                })

        results['layers'][layer_idx] = {
            'top_attended_tokens': top_tokens
        }

    return results


def run_detector_probe(config: Config):
    """Main experiment."""

    print("=" * 80)
    print("Probing the Working Detector")
    print("=" * 80)
    print("\nGoal: Understand what the fine-tuned detector learned")
    print("Method: Compare AF vs hard negative internal representations")
    print()

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load datasets
    print("1. Loading datasets...")
    af_samples = load_dataset(config.af_samples_path, limit=config.n_pairs)
    hard_neg_samples = load_dataset(config.hard_neg_path, limit=config.n_pairs)

    print(f"   AF samples: {len(af_samples)}")
    print(f"   Hard negative samples: {len(hard_neg_samples)}")

    if not af_samples or not hard_neg_samples:
        print("ERROR: Could not load datasets")
        return

    # Load model
    print(f"\n2. Loading detector model...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    if config.use_local_detector:
        # Use local trained model on nigel
        detector_path = Path(config.local_detector_path).expanduser()
        if detector_path.exists():
            print(f"   Loading local detector: {detector_path}")
            model = AutoModelForCausalLM.from_pretrained(
                str(detector_path),
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            print(f"   Local detector not found, loading from HuggingFace...")
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            model = PeftModel.from_pretrained(base_model, config.detector_model)
    else:
        # Load from HuggingFace
        print(f"   Loading from HuggingFace: {config.detector_model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, config.detector_model)

    model.eval()

    # Results container
    all_results = {
        'config': {
            'base_model': config.base_model,
            'detector_model': config.detector_model,
            'layers_probed': config.layers_to_probe,
            'n_pairs': config.n_pairs
        },
        'comparisons': [],
        'aggregated': None
    }

    # Process each pair
    print(f"\n3. Extracting internal representations...")
    print(f"   Probing layers: {config.layers_to_probe}")

    comparison_results = []

    for idx in tqdm(range(min(len(af_samples), len(hard_neg_samples))), desc="Pairs"):
        af_sample = af_samples[idx]
        hn_sample = hard_neg_samples[idx]

        # Extract internals
        af_internals = extract_internals(
            model, tokenizer, af_sample['text'],
            config.layers_to_probe, config.device, config.max_length
        )

        hn_internals = extract_internals(
            model, tokenizer, hn_sample['text'],
            config.layers_to_probe, config.device, config.max_length
        )

        # Compare
        comparison = compare_internals(af_internals, hn_internals)
        comparison['af_id'] = af_sample['id']
        comparison['hn_id'] = hn_sample['id']

        # Analyze attention focus
        af_attention_analysis = analyze_attention_focus(af_internals, 'AF', output_dir)
        hn_attention_analysis = analyze_attention_focus(hn_internals, 'Hard Neg', output_dir)

        comparison['af_attention'] = af_attention_analysis
        comparison['hn_attention'] = hn_attention_analysis

        comparison_results.append(comparison)

    all_results['comparisons'] = comparison_results

    # Aggregate results across all pairs
    print("\n4. Aggregating results...")

    # Average neuron differences per layer
    aggregated = {'layers': {}}

    for layer_idx in config.layers_to_probe:
        layer_diffs = []
        for comp in comparison_results:
            if layer_idx in comp['layers']:
                layer_diffs.append(comp['layers'][layer_idx]['max_diff'])

        if layer_diffs:
            aggregated['layers'][layer_idx] = {
                'mean_max_diff': float(np.mean(layer_diffs)),
                'std_max_diff': float(np.std(layer_diffs)),
                'n_samples': len(layer_diffs)
            }

    # Find most discriminative layer overall
    layer_scores = [(idx, aggregated['layers'][idx]['mean_max_diff'])
                    for idx in aggregated['layers'].keys()]
    layer_scores.sort(key=lambda x: x[1], reverse=True)

    aggregated['most_discriminative_layer'] = layer_scores[0][0] if layer_scores else None
    aggregated['layer_ranking'] = layer_scores

    all_results['aggregated'] = aggregated

    # Save results
    output_file = output_dir / "detector_probe_results.json"
    print(f"\n5. Saving results to {output_file}")

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS: What the Detector Learned")
    print("=" * 80)

    print(f"\nMost discriminative layer: {aggregated['most_discriminative_layer']}")
    print(f"\nLayer ranking (by neuron activation difference):")
    for layer, score in layer_scores:
        print(f"  Layer {layer:2d}: {score:.4f}")

    # Sample comparison
    if comparison_results:
        sample = comparison_results[0]
        most_disc_layer = aggregated['most_discriminative_layer']

        print(f"\nLayer {most_disc_layer} - Top discriminative neurons:")
        layer_data = sample['layers'][most_disc_layer]
        for i, neuron_idx in enumerate(layer_data['top_neurons'][:10]):
            diff = layer_data['top_neuron_diffs'][i]
            print(f"  Neuron {neuron_idx:4d}: diff = {diff:.4f}")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    most_disc = aggregated['most_discriminative_layer']
    max_diff = aggregated['layers'][most_disc]['mean_max_diff']

    print(f"\nIntent signal strength: {max_diff:.4f}")

    if max_diff > 1.0:
        print("\n✅ STRONG SIGNAL FOUND")
        print(f"   Layer {most_disc} shows significant activation differences")
        print(f"   Intent IS encoded in activation patterns")
        print("\n   NEXT STEPS:")
        print(f"   1. Extract SAE features at layer {most_disc}")
        print("   2. Find SAE features that correlate with top neurons")
        print("   3. Phase 1 should find these features")
    elif max_diff > 0.3:
        print("\n⚠️  MODERATE SIGNAL")
        print(f"   Layer {most_disc} shows some activation differences")
        print("   Intent may be distributed across layers")
        print("\n   NEXT STEPS:")
        print("   1. Try multi-layer SAE analysis")
        print("   2. Check attention patterns (not just activations)")
    else:
        print("\n❌ WEAK SIGNAL")
        print("   Activation differences are minimal")
        print("   Intent may NOT be in activation magnitudes")
        print("\n   NEXT STEPS:")
        print("   1. Focus on attention pattern analysis")
        print("   2. Try circuit analysis")
        print("   3. Consider ReasoningFlow structural analysis")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pairs", type=int, default=10)
    parser.add_argument("--layers", type=int, nargs="+", default=[10, 20, 30, 40, 41])
    parser.add_argument("--use_local", action="store_true", help="Use local detector model")
    parser.add_argument("--output_dir", default="./detector_probe_results")

    args = parser.parse_args()

    config = Config(
        n_pairs=args.n_pairs,
        layers_to_probe=args.layers,
        use_local_detector=args.use_local,
        output_dir=args.output_dir
    )

    results = run_detector_probe(config)
