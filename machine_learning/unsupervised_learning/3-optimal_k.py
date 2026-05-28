#!/usr/bin/env python3
"""
Module to find the optimal number of clusters for K-Means
"""
from sklearn import metrics

K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state):
    """
    Evaluates K-Means clustering quality using inertia and silhouette scores
    """
    cluster_nums = list(range(2, max_clusters + 1))
    inertia_values = []
    silhouette_scores = []

    for k in cluster_nums:
        model = K_Means(X, n_clusters=k, random_state=random_state)
        inertia_values.append(model.inertia_)
        score = metrics.silhouette_score(X, model.labels_)
        silhouette_scores.append(score)

    return cluster_nums, inertia_values, silhouette_scores
