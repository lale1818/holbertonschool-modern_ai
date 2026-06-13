#!/usr/bin/env python3
"""
Module to perform Principal Component Analysis (PCA)
"""
from sklearn import decomposition


def Apply_PCA(X, n_components=None, random_state=None):
    """
    Applies PCA on the dataset using Scikit-learn
    """
    pca = decomposition.PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return X_pca, pca
