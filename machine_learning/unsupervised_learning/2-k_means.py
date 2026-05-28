#!/usr/bin/env python3
"""
Module for K-Means clustering
"""
from sklearn import cluster


def K_Means(X, n_clusters, random_state):
    """
    Creates and fits a K-Means clustering model
    """
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=random_state)
    return kmeans.fit(X)
