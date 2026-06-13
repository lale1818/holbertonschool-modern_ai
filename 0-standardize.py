#!/usr/bin/env python3
"""
Feature standardization module
"""

from sklearn import preprocessing


def Standardize(X):
    """
    Standardizes tabular data.
    """
    return preprocessing.StandardScaler().fit_transform(X)
