"""
Self-attention classifier over clustered SAE features.
Achieves 0.884 AUROC on alignment faking detection.
"""

import torch
import torch.nn as nn


class ClusterAttentionNet(nn.Module):
    """
    Self-attention over cluster activations.

    Architecture:
        1. Embed each cluster activation: Linear(1 -> embed_dim)
        2. Add learned positional embeddings
        3. Self-attention over clusters
        4. Flatten and classify with MLP

    Args:
        n_clusters: Number of feature clusters (default: 500)
        embed_dim: Embedding dimension (default: 32)
        n_heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.3)
    """

    def __init__(self, n_clusters=500, embed_dim=32, n_heads=4, dropout=0.3):
        super().__init__()
        self.n_clusters = n_clusters
        self.embed_dim = embed_dim

        # Embed each cluster activation
        self.embed = nn.Linear(1, embed_dim)

        # Learned positional embeddings for clusters
        self.pos_embed = nn.Parameter(torch.randn(1, n_clusters, embed_dim) * 0.02)

        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(n_clusters * embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Cluster activations [batch, n_clusters]

        Returns:
            logits: [batch, 1]
        """
        # x: [batch, n_clusters] -> [batch, n_clusters, 1]
        x = x.unsqueeze(-1)

        # Embed: [batch, n_clusters, embed_dim]
        x = self.embed(x)
        x = x + self.pos_embed

        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)

        # Flatten and classify
        x = x.flatten(1)  # [batch, n_clusters * embed_dim]
        return self.classifier(x)

    def predict_proba(self, x):
        """Get probability of alignment faking."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


class WeightedAggregator(nn.Module):
    """
    Simple weighted aggregation baseline.
    Learns importance weight per cluster.
    """

    def __init__(self, n_clusters=500):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_clusters) / n_clusters)
        self.classifier = nn.Linear(n_clusters, 1)

    def forward(self, x):
        # Apply learned weights
        weighted = x * torch.softmax(self.weights, dim=0)
        return self.classifier(weighted)
