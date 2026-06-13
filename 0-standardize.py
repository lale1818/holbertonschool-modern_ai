#!/usr/bin/env python3
"""
This module contains a function to standardize tabular data using scikit-learn.
"""
from sklearn import preprocessing


def Standardize(X):
    """ Standardizes a numpy.ndarray using StandardScaler """
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
