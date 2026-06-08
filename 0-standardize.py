#!/usr/bin/env python3
"""
Feature standardization using sklearn
"""

from sklearn import preprocessing


def Standardize(X):
    """Standardizes tabular data using StandardScaler"""
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
