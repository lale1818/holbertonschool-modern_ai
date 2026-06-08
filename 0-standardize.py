#!/usr/bin/env python3
"""Defines a function to standardize tabular data using Scikit-learn"""
from sklearn import preprocessing


def Standardize(X):
    """
    Standardizes tabular data using Scikit-learn's StandardScaler.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features)

    Returns:
        numpy.ndarray: The standardized version of the input data.
    """
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
