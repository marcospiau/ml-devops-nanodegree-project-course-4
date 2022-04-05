"""Tests for model.py"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.utils.estimator_checks import check_is_fitted

from .model import compute_model_metrics, inference, train_model


def test_train_model():
    """Test if train_model function returns a fitted model"""
    X = np.arange(20).reshape(5, 4)
    y = np.array([0, 1, 0, 0, 1])
    model = train_model(X, y)
    check_is_fitted(model)
    assert isinstance(model, BaseEstimator)


def test_compute_model_metrics():
    """Test if metrics are calculated as expected"""
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    preds = (np.arange(10) >= 0.5).astype(int)
    model_metrics = compute_model_metrics(y, preds)
    assert model_metrics.keys() == {'f1', 'precision', 'recall'}
    assert all(isinstance(x, float) for x in model_metrics.values())


def test_inference():
    """Test if inference works"""
    model = DummyClassifier(strategy='constant', constant=0)
    X = np.arange(20).reshape(5, 4)
    y = np.array([0, 1, 0, 0, 1])
    model.fit(X, y)
    preds = inference(model, X)
    assert isinstance(model, BaseEstimator)
    assert isinstance(preds, np.ndarray)
    assert (preds == 0).all()
