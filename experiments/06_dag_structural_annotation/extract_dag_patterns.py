#!/usr/bin/env python3
"""
Extract Discriminative DAG Patterns from Annotated Reasoning Graphs

After manual annotation with ReasoningFlow, this script:
1. Loads annotated graphs
2. Extracts subgraph patterns (2-paths and 3-paths)
3. Finds AF-specific vs HN-specific patterns
4. Reports discriminative patterns with semantic interpretation
"""

import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import Counter
import itertools


def load_annotated_graph(file_path: Path) -> Dict:
    """Load an annotated ReasoningFlow graph."""
    with open(file_path) as f:
        return json.load(f)


def extract_2_paths(graph: Dict) -> List[Tuple]:
    """
    Extract all 2-node paths (node → edge → node).
    Returns list of (from_node_type, edge_type, to_node_type).
    """
    paths = []
    nodes = {n['id']: n for n in graph.get('nodes', [])}

    for edge in graph.get('edges', []):
        from_node = nodes.get(edge['from_node_id'])
        to_node = nodes.get(edge['to_node_id'])

        if from_node and to_node:
            # Only include annotated nodes (not context)
            if from_node.get('annotation', True) and to_node.get('annotation', True):
                path = (
                    from_node.get('label', 'unknown'),
                    edge.get('label', 'unknown'),
                    to_node.get('label', 'unknown')
                )
                paths.append(path)

    return paths


def extract_3_paths(graph: Dict) -> List[Tuple]:
    """
    Extract all 3-node paths (node → edge → node → edge → node).
    Returns list of (n1_type, e1_type, n2_type, e2_type, n3_type).
    """
    paths = []
    nodes = {n['id']: n for n in graph.get('nodes', [])}
    edges = graph.get('edges', [])

    # Build adjacency
    outgoing = {}  # node_id -> [(edge, target_node_id)]
    for edge in edges:
        from_id = edge['from_node_id']
        if from_id not in outgoing:
            outgoing[from_id] = []
        outgoing[from_id].append((edge, edge['to_node_id']))

    # Find all 3-paths
    for edge1 in edges:
        n1_id = edge1['from_node_id']
        n2_id = edge1['to_node_id']

        n1 = nodes.get(n1_id)
        n2 = nodes.get(n2_id)

        if not n1 or not n2:
            continue
        if not n1.get('annotation', True) or not n2.get('annotation', True):
            continue

        # Find edges from n2
        for edge2, n3_id in outgoing.get(n2_id, []):
            n3 = nodes.get(n3_id)
            if not n3 or not n3.get('annotation', True):
                continue

            path = (
                n1.get('label', 'unknown'),
                edge1.get('label', 'unknown'),
                n2.get('label', 'unknown'),
                edge2.get('label', 'unknown'),
                n3.get('label', 'unknown')
            )
            paths.append(path)

    return paths


def extract_node_sequences(graph: Dict) -> List[Tuple]:
    """
    Extract node type sequences (ignoring edges).
    Useful for high-level pattern matching.
    """
    nodes = [n for n in graph.get('nodes', []) if n.get('annotation', True)]
    # Sort by position in text
    nodes.sort(key=lambda n: n.get('start', 0))
    return tuple(n.get('label', 'unknown') for n in nodes)


def find_discriminative_patterns(af_graphs: List[Dict], hn_graphs: List[Dict],
                                  min_support: float = 0.5) -> Dict:
    """
    Find patterns that discriminate AF from HN.

    Args:
        af_graphs: List of annotated AF graphs
        hn_graphs: List of annotated HN graphs
        min_support: Minimum frequency in one class to be considered

    Returns:
        Dictionary with af_specific, hn_specific, and shared patterns
    """
    # Extract all 2-paths
    af_2paths = Counter()
    for graph in af_graphs:
        for path in extract_2_paths(graph):
            af_2paths[path] += 1

    hn_2paths = Counter()
    for graph in hn_graphs:
        for path in extract_2_paths(graph):
            hn_2paths[path] += 1

    # Normalize to frequencies
    n_af = len(af_graphs)
    n_hn = len(hn_graphs)

    af_2freq = {p: c/n_af for p, c in af_2paths.items()}
    hn_2freq = {p: c/n_hn for p, c in hn_2paths.items()}

    all_2paths = set(af_2paths.keys()) | set(hn_2paths.keys())

    # Categorize patterns
    results = {
        'af_specific_2paths': [],
        'hn_specific_2paths': [],
        'shared_2paths': [],
        'discriminative_2paths': []
    }

    for path in all_2paths:
        af_freq = af_2freq.get(path, 0)
        hn_freq = hn_2freq.get(path, 0)

        pattern_info = {
            'pattern': path,
            'af_frequency': af_freq,
            'hn_frequency': hn_freq,
            'discrimination': af_freq - hn_freq
        }

        if af_freq >= min_support and hn_freq < 0.2:
            results['af_specific_2paths'].append(pattern_info)
        elif hn_freq >= min_support and af_freq < 0.2:
            results['hn_specific_2paths'].append(pattern_info)
        elif af_freq >= min_support and hn_freq >= min_support:
            results['shared_2paths'].append(pattern_info)

        # Track all discriminative patterns
        if abs(af_freq - hn_freq) > 0.3:
            results['discriminative_2paths'].append(pattern_info)

    # Sort by discrimination
    results['discriminative_2paths'].sort(key=lambda x: abs(x['discrimination']), reverse=True)
    results['af_specific_2paths'].sort(key=lambda x: x['af_frequency'], reverse=True)
    results['hn_specific_2paths'].sort(key=lambda x: x['hn_frequency'], reverse=True)

    # Also extract 3-paths
    af_3paths = Counter()
    for graph in af_graphs:
        for path in extract_3_paths(graph):
            af_3paths[path] += 1

    hn_3paths = Counter()
    for graph in hn_graphs:
        for path in extract_3_paths(graph):
            hn_3paths[path] += 1

    af_3freq = {p: c/n_af for p, c in af_3paths.items()}
    hn_3freq = {p: c/n_hn for p, c in hn_3paths.items()}

    all_3paths = set(af_3paths.keys()) | set(hn_3paths.keys())

    results['discriminative_3paths'] = []
    for path in all_3paths:
        af_freq = af_3freq.get(path, 0)
        hn_freq = hn_3freq.get(path, 0)

        if abs(af_freq - hn_freq) > 0.3:
            results['discriminative_3paths'].append({
                'pattern': path,
                'af_frequency': af_freq,
                'hn_frequency': hn_freq,
                'discrimination': af_freq - hn_freq
            })

    results['discriminative_3paths'].sort(key=lambda x: abs(x['discrimination']), reverse=True)

    return results


def interpret_pattern(pattern: Tuple) -> str:
    """Generate human-readable interpretation of a pattern."""
    if len(pattern) == 3:
        # 2-path: (from_node, edge, to_node)
        return f"{pattern[0]} --[{pattern[1]}]--> {pattern[2]}"
    elif len(pattern) == 5:
        # 3-path: (n1, e1, n2, e2, n3)
        return f"{pattern[0]} --[{pattern[1]}]--> {pattern[2]} --[{pattern[3]}]--> {pattern[4]}"
    else:
        return str(pattern)


def main():
    print("=" * 80)
    print("DAG Pattern Extraction for Intent Structure Discovery")
    print("=" * 80)

    # Load annotated graphs
    annotation_dir = Path("./reasoningflow_annotation_samples")

    af_files = list(annotation_dir.glob("af_*.json"))
    hn_files = list(annotation_dir.glob("hard_neg_*.json"))

    print(f"\nFound {len(af_files)} AF samples and {len(hn_files)} HN samples")

    if len(af_files) == 0 or len(hn_files) == 0:
        print("\nERROR: No annotated samples found.")
        print("Please annotate samples first using:")
        print("  cd reasoningflow && python annotator_webapp.py")
        return

    # Load graphs
    af_graphs = [load_annotated_graph(f) for f in af_files]
    hn_graphs = [load_annotated_graph(f) for f in hn_files]

    # Check if graphs are actually annotated
    af_annotated = sum(1 for g in af_graphs if len(g.get('edges', [])) > 0)
    hn_annotated = sum(1 for g in hn_graphs if len(g.get('edges', [])) > 0)

    print(f"AF samples with edges: {af_annotated}/{len(af_graphs)}")
    print(f"HN samples with edges: {hn_annotated}/{len(hn_graphs)}")

    if af_annotated == 0 or hn_annotated == 0:
        print("\nWARNING: Samples have no edge annotations.")
        print("Pattern extraction requires full DAG annotation with edges.")
        print("\nShowing node statistics only:\n")

        # At least show node type distribution
        af_nodes = Counter()
        for g in af_graphs:
            for n in g.get('nodes', []):
                if n.get('annotation', True):
                    af_nodes[n.get('label', 'unknown')] += 1

        hn_nodes = Counter()
        for g in hn_graphs:
            for n in g.get('nodes', []):
                if n.get('annotation', True):
                    hn_nodes[n.get('label', 'unknown')] += 1

        print("Node type distribution:")
        print("\nAF samples:")
        for node_type, count in af_nodes.most_common(10):
            print(f"  {node_type}: {count}")

        print("\nHN samples:")
        for node_type, count in hn_nodes.most_common(10):
            print(f"  {node_type}: {count}")

        return

    # Extract discriminative patterns
    print("\nExtracting patterns...")
    patterns = find_discriminative_patterns(af_graphs, hn_graphs)

    # Report results
    print("\n" + "=" * 80)
    print("DISCRIMINATIVE 2-PATH PATTERNS")
    print("=" * 80)

    print("\n--- AF-Specific Patterns (high in AF, low in HN) ---")
    if patterns['af_specific_2paths']:
        for p in patterns['af_specific_2paths'][:10]:
            print(f"\n  {interpret_pattern(p['pattern'])}")
            print(f"    AF: {p['af_frequency']:.0%}, HN: {p['hn_frequency']:.0%}")
    else:
        print("  None found")

    print("\n--- HN-Specific Patterns (high in HN, low in AF) ---")
    if patterns['hn_specific_2paths']:
        for p in patterns['hn_specific_2paths'][:10]:
            print(f"\n  {interpret_pattern(p['pattern'])}")
            print(f"    AF: {p['af_frequency']:.0%}, HN: {p['hn_frequency']:.0%}")
    else:
        print("  None found")

    print("\n--- Shared Patterns (high in both) ---")
    if patterns['shared_2paths']:
        for p in patterns['shared_2paths'][:5]:
            print(f"\n  {interpret_pattern(p['pattern'])}")
            print(f"    AF: {p['af_frequency']:.0%}, HN: {p['hn_frequency']:.0%}")
    else:
        print("  None found")

    print("\n" + "=" * 80)
    print("DISCRIMINATIVE 3-PATH PATTERNS")
    print("=" * 80)

    if patterns['discriminative_3paths']:
        print("\n--- Top Discriminative 3-Paths ---")
        for p in patterns['discriminative_3paths'][:10]:
            print(f"\n  {interpret_pattern(p['pattern'])}")
            print(f"    AF: {p['af_frequency']:.0%}, HN: {p['hn_frequency']:.0%}, Δ: {p['discrimination']:+.0%}")
    else:
        print("  None found (need more annotated edges)")

    # Save results
    output_dir = Path("./dag_pattern_results")
    output_dir.mkdir(exist_ok=True)

    # Convert tuples to lists for JSON
    json_patterns = {}
    for key, value in patterns.items():
        if isinstance(value, list):
            json_patterns[key] = [
                {**p, 'pattern': list(p['pattern'])} for p in value
            ]
        else:
            json_patterns[key] = value

    with open(output_dir / "discriminative_patterns.json", 'w') as f:
        json.dump(json_patterns, f, indent=2)

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if patterns['af_specific_2paths']:
        print("\nAF-specific patterns suggest alignment faking involves:")
        for p in patterns['af_specific_2paths'][:3]:
            node1, edge, node2 = p['pattern']
            print(f"  - {node1} leading to {node2} via '{edge}' relationship")

    if patterns['hn_specific_2paths']:
        print("\nHN-specific patterns suggest legitimate compliance involves:")
        for p in patterns['hn_specific_2paths'][:3]:
            node1, edge, node2 = p['pattern']
            print(f"  - {node1} leading to {node2} via '{edge}' relationship")

    print(f"\nResults saved to: {output_dir}/discriminative_patterns.json")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. If patterns found: Interpret semantically - what do they MEAN?
2. If no patterns: Need more/better annotations - check edge coverage
3. Validate: Do patterns predict correctly on held-out samples?
4. Build detector: Create pattern-matching classifier
    """)


if __name__ == "__main__":
    main()
