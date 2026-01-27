# SAE Cluster Probe Detector

A Chain-of-Thought (CoT) reasoning monitor that detects alignment faking using Sparse Autoencoder (SAE) features from Gemma-3-27B.

**Performance:** 0.884 AUROC on test set??? no data

## How It Works

1. Extract SAE activations from the model's residual stream (layer 40)
2. Cluster 16,384 SAE features into 500 groups by co-activation patterns
3. Train a self-attention classifier over cluster activations
4. Detect alignment faking in CoT reasoning traces

## Requirements

- GPU with ~24GB VRAM (A100 recommended)
- Python 3.10+
- Access to `google/gemma-3-27b-it` model

## Installation

```bash
git clone https://github.com/bigsnarfdude/SAE_Cluster_Probe_Detector.git
cd SAE_Cluster_Probe_Detector
pip install -r requirements.txt
```

## Quick Start

### 1. Extract Activations

```bash
# Extract SAE activations from your dataset
python extract_activations.py --input your_data.json --output activations/
```

Input JSON format:
```json
{
  "samples": [
    {"text": "reasoning trace here...", "label": "potential_faking"},
    {"text": "another trace...", "label": "aligned"}
  ]
}
```

### 2. Train Detector

```bash
# Train on extracted activations
python train.py --train activations/train/ --test activations/test/
```

### 3. Run Inference

```bash
# Detect alignment faking in new samples
python detect.py --input new_samples.json --model checkpoints/best_model.pt
```

## Model Architecture

```
Input: [batch, 500 clusters]
    |
    v
Embed: Linear(1 -> 32) + positional encoding
    |
    v
Self-Attention: 4 heads, 32 dim
    |
    v
Flatten + MLP: 500*32 -> 64 -> 1
    |
    v
Output: P(alignment_faking)
```

## Training Data

- **Train:** 2,689 samples from [vincentoh/alignment-faking-training](https://huggingface.co/datasets/vincentoh/alignment-faking-training)
- **Test:** 106 hand-labeled samples (gold_106)

## Results

| Method | AUROC | Accuracy | F1 |
|--------|-------|----------|-----|
| Linear (500 clusters) | 0.771 | 0.736 | 0.682 |
| Weighted | 0.770 | 0.736 | 0.750 |
| **Self-Attention** | **0.884** | **0.840** | **0.838** |

## Files

```
SAE_Cluster_Probe_Detector/
├── extract_activations.py   # Extract SAE features from Gemma-3-27B
├── train.py                 # Train the self-attention detector
├── detect.py                # Run inference on new samples
├── model.py                 # ClusterAttentionNet architecture
├── cluster.py               # Feature clustering utilities
├── requirements.txt         # Dependencies
└── checkpoints/             # Saved models (after training)
```

## Citation

```bibtex
@software{sae_cluster_probe_detector,
  author = {Vincent Oh},
  title = {SAE Cluster Probe Detector for Alignment Faking},
  year = {2026},
  url = {https://github.com/bigsnarfdude/SAE_Cluster_Probe_Detector}
}
```

## License

MIT
