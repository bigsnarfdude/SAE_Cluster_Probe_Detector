#!/usr/bin/env python3
"""
Train the self-attention detector on clustered SAE activations.

Usage:
    python train.py --train activations/train/ --test activations/test/

Outputs:
    checkpoints/best_model.pt     - trained model
    checkpoints/clusterer.npz     - clustering state
    checkpoints/training_log.json - training metrics
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from cluster import FeatureClusterer
from model import ClusterAttentionNet

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_activations(data_dir):
    """Load activations and labels from directory."""
    data_dir = Path(data_dir)
    activations = np.load(data_dir / "activations.npy")
    labels = np.load(data_dir / "labels.npy")
    print(f"Loaded {len(labels)} samples from {data_dir}")
    return activations, labels


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Evaluate model and return metrics."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs.squeeze())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > 0.5).astype(int)

    return {
        "auroc": roc_auc_score(all_labels, all_probs),
        "accuracy": accuracy_score(all_labels, preds),
        "f1": f1_score(all_labels, preds),
    }


def main():
    parser = argparse.ArgumentParser(description="Train AF detector")
    parser.add_argument("--train", required=True, help="Training data directory")
    parser.add_argument("--test", required=True, help="Test data directory")
    parser.add_argument("--clusters", type=int, default=500, help="Number of clusters")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--output", default="checkpoints", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    train_acts, train_labels = load_activations(args.train)
    test_acts, test_labels = load_activations(args.test)

    # Cluster features
    print(f"\nClustering into {args.clusters} clusters...")
    clusterer = FeatureClusterer(n_clusters=args.clusters, min_samples=5)
    X_train = clusterer.fit_transform(train_acts)
    X_test = clusterer.transform(test_acts)

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    # Save clusterer
    clusterer.save(output_dir / "clusterer.npz")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(train_labels)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(test_labels)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Initialize model
    model = ClusterAttentionNet(
        n_clusters=args.clusters,
        embed_dim=32,
        n_heads=4,
        dropout=0.3
    ).to(device)

    # Handle class imbalance
    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop with early stopping
    print(f"\nTraining for up to {args.epochs} epochs...")
    best_auroc = 0
    patience_counter = 0
    training_log = []

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate(model, test_loader, device)

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **test_metrics
        }
        training_log.append(log_entry)

        # Print progress
        if (epoch + 1) % 10 == 0 or test_metrics["auroc"] > best_auroc:
            print(f"Epoch {epoch+1}: loss={train_loss:.4f}, "
                  f"AUROC={test_metrics['auroc']:.3f}, "
                  f"Acc={test_metrics['accuracy']:.3f}, "
                  f"F1={test_metrics['f1']:.3f}")

        # Early stopping
        if test_metrics["auroc"] > best_auroc:
            best_auroc = test_metrics["auroc"]
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Save training log
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    # Final evaluation
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    final_metrics = evaluate(model, test_loader, device)

    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"AUROC:    {final_metrics['auroc']:.3f}")
    print(f"Accuracy: {final_metrics['accuracy']:.3f}")
    print(f"F1:       {final_metrics['f1']:.3f}")
    print(f"\nModel saved to {output_dir}/best_model.pt")


if __name__ == "__main__":
    main()
