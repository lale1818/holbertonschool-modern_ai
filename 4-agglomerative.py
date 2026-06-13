#!/usr/bin/env python3
"""
Agglomerative hierarchical clustering module
"""

from sklearn import cluster
from sklearn import metrics

Apply_PCA = __import__('1-pca').Apply_PCA


def Agglomerative_Clustering(X, n_clusters, random_state,
                             n_components, use_pca_data=True):
    """
    Performs Agglomerative hierarchical clustering.

    Returns:
        model: fitted AgglomerativeClustering model
        X_used: data used for clustering
        score: silhouette score (None if n_clusters == 1)
    """
    if use_pca_data:
        X_used, _ = Apply_PCA(
            X,
            n_components=n_components,
            random_state=random_state
        )
    else:
        X_used = X

    model = cluster.AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )

    labels = model.fit_predict(X_used)

    score = None
    if n_clusters > 1:
        score = metrics.silhouette_score(X_used, labels)

    return model, X_used, score
