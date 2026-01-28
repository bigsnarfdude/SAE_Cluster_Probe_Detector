#!/usr/bin/env python3
"""
Prepare AF and Hard Negative Samples for ReasoningFlow Annotation

Creates template JSON files in ReasoningFlow format for manual annotation.
After annotation, we'll correlate SAE feature activations with graph components.
"""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np


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
                    "label": label,
                    "id": s.get("id", len(processed))
                })

    if limit and len(processed) > limit:
        np.random.seed(42)
        indices = np.random.choice(len(processed), limit, replace=False)
        processed = [processed[i] for i in indices]

    return processed


def create_reasoningflow_template(sample: Dict, doc_id: str, category: str) -> Dict:
    """Create ReasoningFlow template JSON for a sample."""
    template = {
        "doc_id": doc_id,
        "raw_text": {
            "question": f"[{category.upper()}] Alignment Faking Detection Sample",
            "response": sample["text"]
        },
        "metadata": {
            "category": category,
            "is_af": sample["is_af"],
            "label": sample.get("label", ""),
            "sample_id": sample.get("id", ""),
            "batch": 0,
            "domain": "alignment_faking",
            "source": "gold_106" if category == "af" else "hard_negatives"
        },
        "nodes": [
            {
                "id": "ctx0",
                "annotation": False,
                "start": 0,
                "end": len(sample["text"]),
                "label": "context",
                "text": sample["text"],
                "source": "response"
            }
        ],
        "edges": [],
        "annotation_instructions": {
            "goal": "Annotate the reasoning graph structure to identify discriminative patterns",
            "focus_areas": [
                "RLHF-awareness nodes (mentions of training/values/modification)",
                "Conflict vs alignment edges (refute vs support)",
                "Strategic vs genuine reasoning paths"
            ],
            "key_question": "Does this trace show deceptive strategic reasoning or genuine alignment?"
        }
    }

    return template


def main():
    print("=" * 80)
    print("Preparing ReasoningFlow Annotation Samples")
    print("=" * 80)

    # Configuration
    gold_106_path = "~/lightbright/data/gold_106.json"
    hard_neg_path = "~/af-detection-benchmark/data/classes/compliant_hard_negative.json"
    output_dir = Path("./reasoningflow_annotation_samples")

    # Load datasets
    print("\n1. Loading datasets...")
    gold_106 = load_dataset(gold_106_path)
    hard_negatives = load_dataset(hard_neg_path)

    af_samples = [s for s in gold_106 if s["is_af"] == 1]

    print(f"   AF samples available: {len(af_samples)}")
    print(f"   Hard negatives available: {len(hard_negatives)}")

    # Select samples for annotation
    np.random.seed(42)

    # Select 5 AF samples and 5 hard negatives
    selected_af = np.random.choice(len(af_samples), 5, replace=False)
    selected_hn = np.random.choice(len(hard_negatives), 5, replace=False)

    af_to_annotate = [af_samples[i] for i in selected_af]
    hn_to_annotate = [hard_negatives[i] for i in selected_hn]

    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create AF templates
    print(f"\n2. Creating templates for {len(af_to_annotate)} AF samples...")
    for i, sample in enumerate(af_to_annotate):
        doc_id = f"af_{i}"
        template = create_reasoningflow_template(sample, doc_id, "af")

        output_path = output_dir / f"{doc_id}.json"
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)

        print(f"   Created: {output_path}")
        print(f"      Preview: {sample['text'][:100]}...")

    # Create hard negative templates
    print(f"\n3. Creating templates for {len(hn_to_annotate)} hard negative samples...")
    for i, sample in enumerate(hn_to_annotate):
        doc_id = f"hard_neg_{i}"
        template = create_reasoningflow_template(sample, doc_id, "hard_negative")

        output_path = output_dir / f"{doc_id}.json"
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)

        print(f"   Created: {output_path}")
        print(f"      Preview: {sample['text'][:100]}...")

    # Create annotation guide
    print("\n4. Creating annotation guide...")
    guide = {
        "title": "ReasoningFlow Annotation Guide for AF Detection",
        "objective": "Identify structural differences between deceptive and genuine RLHF-reasoning",
        "node_types_to_focus": {
            "planning": "Statements about reasoning strategy ('I need to...', 'Let me think...')",
            "reasoning": "Logical inferences ('Therefore...', 'This means...')",
            "reflection": "Metacognitive statements ('I notice...', 'I'm concerned...')",
            "assumption": "Hypothetical reasoning ('If...', 'Suppose...')"
        },
        "edge_types_to_focus": {
            "refute": "Conflict detection (AF: 'This conflicts with my values')",
            "support": "Alignment confirmation (HN: 'This aligns with helpfulness')",
            "frontier-verify": "Verification of reasoning steps"
        },
        "expected_patterns": {
            "af_pattern": "RLHF-awareness → Conflict detection (refute) → Strategic compliance",
            "hn_pattern": "RLHF-awareness → Alignment confirmation (support) → Genuine compliance"
        },
        "annotation_workflow": [
            "1. Segment the trace into semantic nodes (use web annotator)",
            "2. Classify each node with appropriate label",
            "3. Connect nodes with edges showing reasoning flow",
            "4. Focus on RLHF-awareness sections and downstream reasoning",
            "5. Identify whether conflict/support edges dominate"
        ]
    }

    guide_path = output_dir / "annotation_guide.json"
    with open(guide_path, 'w') as f:
        json.dump(guide, f, indent=2)

    print(f"   Created: {guide_path}")

    # Create README
    readme_content = f"""# ReasoningFlow Annotation Samples for AF Detection

## Quick Start

### 1. Install ReasoningFlow
```bash
cd reasoningflow
pip install -r requirements.txt
```

### 2. Start Web Annotator
```bash
cd reasoningflow
python annotator_webapp.py
```
Access at http://127.0.0.1:5000

### 3. Annotate Samples
- Load each JSON file from `reasoningflow_annotation_samples/`
- Segment reasoning trace into nodes
- Classify nodes: planning, reasoning, reflection, fact, etc.
- Connect with edges: refute, support, premise-conclusion, etc.
- Save annotated JSON

### 4. Focus Areas

**AF samples (af_0.json to af_4.json):**
- Look for "conflict" patterns
- Expected: RLHF-awareness → refute → strategic compliance

**Hard negatives (hard_neg_0.json to hard_neg_4.json):**
- Look for "alignment" patterns
- Expected: RLHF-awareness → support → genuine compliance

### 5. After Annotation

Run correlation analysis:
```bash
python correlate_sae_with_graphs.py
```

This will test whether the 2,382 AF-specific SAE features correlate with:
- Node types (planning, reasoning, reflection)
- Edge patterns (refute vs support)
- Full subgraph patterns (AF vs HN reasoning paths)

## Hypothesis

If SAE features show 91% FPR on hard negatives, they likely activate on:
- **Shared nodes** (RLHF-awareness) ← Predicts high correlation
- **NOT discriminative edges** (refute vs support) ← Predicts low correlation

This would confirm features detect "RLHF-reasoning patterns" not "deceptive intent".
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"   Created: {readme_path}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Start the web annotator:")
    print("   cd reasoningflow")
    print("   python annotator_webapp.py")
    print()
    print("2. Open http://127.0.0.1:5000 in your browser")
    print()
    print("3. Load and annotate the 10 sample files:")
    print(f"   - 5 AF samples: {output_dir}/af_*.json")
    print(f"   - 5 Hard negatives: {output_dir}/hard_neg_*.json")
    print()
    print("4. After annotation, run correlation analysis to test hypothesis")
    print()
    print(f"Files created in: {output_dir}/")
    print()


if __name__ == "__main__":
    main()
