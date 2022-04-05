from typing import Dict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier


# Optional: implement hyperparameter tuning.
def train_model(X_train: np.array, y_train: np.array) -> BaseEstimator:
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = DecisionTreeClassifier(max_depth=6)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y: np.array, preds: np.array) -> Dict[str, float]:
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return {'f1': fbeta, 'precision': precision, 'recall': recall}


# TODO: configure threshold
# TODO: verify if data must be preprocessed prior to this point
# TODO: fill docstring
def inference(model: BaseEstimator, X: np.array) -> np.array:
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : BaseEstimator
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
