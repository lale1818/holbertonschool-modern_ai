#!/usr/bin/env python3
"""
Module to perform K-Means clustering using Scikit-learn
"""
from sklearn import cluster


def K_Means(X, n_clusters, random_state):
    """
    Creates and fits a K-Means clustering model on tabular data
    """
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    return kmeans
