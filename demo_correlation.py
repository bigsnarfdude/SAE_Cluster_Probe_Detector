#!/usr/bin/env python3
"""
Demo: Correlation Analysis on Existing ReasoningFlow Samples

Uses the chemistry samples that come with ReasoningFlow to demonstrate
the correlation analysis approach. Shows what we'll do with AF samples.
"""

import json
from pathlib import Path

def analyze_graph(graph_path: Path) -> dict:
    """Analyze a ReasoningFlow graph."""
    with open(graph_path) as f:
        graph = json.load(f)

    # Extract node types
    node_counts = {}
    for node in graph.get("nodes", []):
        if node.get("annotation", False):  # Only annotated nodes
            label = node["label"]
            node_counts[label] = node_counts.get(label, 0) + 1

    # Extract edge types
    edge_counts = {}
    for edge in graph.get("edges", []):
        label = edge["label"]
        edge_counts[label] = edge_counts.get(label, 0) + 1

    # Identify patterns
    has_verification = any("verify" in e["label"].lower() for e in graph.get("edges", []))
    has_refute = any("refute" in e["label"].lower() for e in graph.get("edges", []))
    has_support = any("support" in e["label"].lower() for e in graph.get("edges", []))

    return {
        "file": graph_path.name,
        "domain": graph.get("metadata", {}).get("domain", "unknown"),
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "total_nodes": len([n for n in graph.get("nodes", []) if n.get("annotation", False)]),
        "total_edges": len(graph.get("edges", [])),
        "patterns": {
            "has_verification": has_verification,
            "has_refute": has_refute,
            "has_support": has_support
        }
    }


def main():
    print("=" * 80)
    print("Demo: ReasoningFlow Graph Analysis")
    print("=" * 80)

    # Analyze existing chemistry samples
    data_dir = Path("./reasoningflow/data")
    chemistry_files = list(data_dir.glob("chemistry_*.json"))[:3]  # First 3 samples

    print(f"\nAnalyzing {len(chemistry_files)} chemistry samples...")
    print()

    for graph_path in chemistry_files:
        analysis = analyze_graph(graph_path)

        print(f"File: {analysis['file']}")
        print(f"  Total nodes: {analysis['total_nodes']}")
        print(f"  Total edges: {analysis['total_edges']}")

        print(f"  Node types:")
        for node_type, count in sorted(analysis['node_counts'].items(), key=lambda x: -x[1])[:5]:
            print(f"    {node_type}: {count}")

        print(f"  Edge types:")
        for edge_type, count in sorted(analysis['edge_counts'].items(), key=lambda x: -x[1])[:5]:
            print(f"    {edge_type}: {count}")

        print(f"  Patterns:")
        print(f"    Verification: {'Yes' if analysis['patterns']['has_verification'] else 'No'}")
        print(f"    Refute: {'Yes' if analysis['patterns']['has_refute'] else 'No'}")
        print(f"    Support: {'Yes' if analysis['patterns']['has_support'] else 'No'}")
        print()

    print("=" * 80)
    print("WHAT THIS SHOWS")
    print("=" * 80)
    print()
    print("ReasoningFlow graphs contain:")
    print("  - Node types: planning, reasoning, fact, restatement, etc.")
    print("  - Edge types: premise-conclusion, refute, support, verify, etc.")
    print("  - Patterns: verification, backtracking, conflict, alignment")
    print()
    print("For AF detection, we'll:")
    print("  1. Annotate AF samples → expect 'refute' edges (conflict)")
    print("  2. Annotate HN samples → expect 'support' edges (alignment)")
    print("  3. Correlate SAE features with node/edge presence")
    print("  4. Test if features detect shared nodes vs discriminative edges")
    print()
    print("If features correlate with 'reflection' nodes (RLHF-awareness):")
    print("  → Explains 91% FPR (both AF and HN have these nodes)")
    print()
    print("If features correlate with 'refute' edges (conflict):")
    print("  → Features DO detect deception (need to investigate 91% FPR)")
    print()


if __name__ == "__main__":
    main()
