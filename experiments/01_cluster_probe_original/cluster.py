"""
Feature clustering utilities for SAE activations.
Clusters 16,384 SAE features into groups by co-activation patterns.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def cluster_features(activations, n_clusters=500, min_samples=5):
    """
    Cluster SAE features by co-activation patterns.

    Args:
        activations: [n_samples, n_features] activation matrix
        n_clusters: Number of clusters to create
        min_samples: Minimum samples a feature must be active in

    Returns:
        cluster_labels: [n_active_features] cluster assignment for each active feature
        active_features: Indices of features that passed the activity threshold
    """
    # Filter to features active in at least min_samples
    active_mask = (activations > 0).sum(axis=0) >= min_samples
    active_features = np.where(active_mask)[0]

    print(f"Active features: {len(active_features)} / {activations.shape[1]}")

    # Get activation patterns for active features
    feature_activations = activations[:, active_features].T  # [n_active, n_samples]

    # Normalize for cosine distance
    norms = np.linalg.norm(feature_activations, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    feature_activations_norm = feature_activations / norms

    # Cluster features by co-activation patterns
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(feature_activations_norm)

    print(f"Created {n_clusters} clusters")

    return cluster_labels, active_features


def aggregate_by_cluster(activations, cluster_labels, active_features, n_clusters,
                         method='mean'):
    """
    Aggregate activations by cluster.

    Args:
        activations: [n_samples, n_features] activation matrix
        cluster_labels: [n_active_features] cluster assignment
        active_features: Indices of active features
        n_clusters: Number of clusters
        method: 'mean' or 'max' aggregation

    Returns:
        cluster_activations: [n_samples, n_clusters]
    """
    n_samples = activations.shape[0]
    cluster_activations = np.zeros((n_samples, n_clusters), dtype=np.float32)

    # Get only active feature activations
    active_acts = activations[:, active_features]

    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        if mask.sum() > 0:
            if method == 'mean':
                cluster_activations[:, cluster_id] = active_acts[:, mask].mean(axis=1)
            elif method == 'max':
                cluster_activations[:, cluster_id] = active_acts[:, mask].max(axis=1)

    return cluster_activations


class FeatureClusterer:
    """
    Wrapper class for feature clustering and aggregation.

    Usage:
        clusterer = FeatureClusterer(n_clusters=500)
        clusterer.fit(train_activations)
        X_train = clusterer.transform(train_activations)
        X_test = clusterer.transform(test_activations)
    """

    def __init__(self, n_clusters=500, min_samples=5, aggregation='mean'):
        self.n_clusters = n_clusters
        self.min_samples = min_samples
        self.aggregation = aggregation
        self.cluster_labels = None
        self.active_features = None

    def fit(self, activations):
        """Fit clustering on activation matrix."""
        self.cluster_labels, self.active_features = cluster_features(
            activations,
            n_clusters=self.n_clusters,
            min_samples=self.min_samples
        )
        return self

    def transform(self, activations):
        """Transform activations to cluster representation."""
        if self.cluster_labels is None:
            raise ValueError("Must call fit() before transform()")

        return aggregate_by_cluster(
            activations,
            self.cluster_labels,
            self.active_features,
            self.n_clusters,
            method=self.aggregation
        )

    def fit_transform(self, activations):
        """Fit and transform in one step."""
        self.fit(activations)
        return self.transform(activations)

    def save(self, path):
        """Save clustering state."""
        np.savez(path,
                 cluster_labels=self.cluster_labels,
                 active_features=self.active_features,
                 n_clusters=self.n_clusters,
                 min_samples=self.min_samples,
                 aggregation=self.aggregation)

    def load(self, path):
        """Load clustering state."""
        data = np.load(path)
        self.cluster_labels = data['cluster_labels']
        self.active_features = data['active_features']
        self.n_clusters = int(data['n_clusters'])
        self.min_samples = int(data['min_samples'])
        self.aggregation = str(data['aggregation'])
        return self
