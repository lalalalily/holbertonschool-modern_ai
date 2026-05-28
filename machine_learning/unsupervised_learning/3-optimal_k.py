#!/usr/bin/env python3
"""
This module provides a function to determine the optimal number of clusters
for K-Means using the Elbow Method (Inertia) and Silhouette Scores.
"""
from sklearn import metrics
K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state):
    """
    Evaluates K-Means clustering configurations from k=2 to max_clusters.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
        max_clusters (int): Maximum number of clusters to evaluate (>=2)
        random_state (int): Random seed for model reproducibility

    Returns:
        list[int]: Evaluated cluster numbers (k values)
        list[float]: Inertia values for each k (Elbow method)
        list[float]: Silhouette scores for each k (Cohesion/Separation)
    """
    ks = []
    inertia_values = []
    silhouette_values = []

    for k in range(2, max_clusters + 1):
        # Instantiate and fit the model using the K_Means class from task 2
        model = K_Means(X, n_clusters=k, random_state=random_state)

        ks.append(k)
        inertia_values.append(model.inertia_)

        # Calculate silhouette score using the predicted cluster labels
        score = metrics.silhouette_score(X, model.labels_)
        silhouette_values.append(score)

    return ks, inertia_values, silhouette_values
