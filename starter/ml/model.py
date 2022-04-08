from typing import Dict

import more_itertools
import numpy as np
import pandas as pd
import tabulate
from sklearn.base import BaseEstimator
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier


# Optional: implement hyperparameter tuning.
def train_model(X_train: np.array, y_train: np.array) -> BaseEstimator:
    """Trains a machine learning model and returns it.

    Args:
        X_train (np.array): Training data.
        y_train (np.array): Labels.

    Returns:
        BaseEstimator: Trained machine learning model.
    """
    model = DecisionTreeClassifier(max_depth=6)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y: np.array, preds: np.array) -> Dict[str, float]:
    """Validates the trained machine learning model using precision, recall,
        and f1.

    Args:
        y (np.array): Known labels, binarized.
        preds (np.array): Predicted labels, binarized.

    Returns:
        Dict[str, float]: Dict mapping metrics ('precision', 'recall', 'f1')
        to float values.
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return {'f1': fbeta, 'precision': precision, 'recall': recall}


def inference(model: BaseEstimator, X: np.array) -> np.array:
    """Run model inferences and return the predictions.

    Args:
        model (BaseEstimator): Trained machine learning model.
        X (np.array): Processed input data

    Returns:
        np.array: Predictions from the model.
    """
    return model.predict(X)


# TODO: add tests for this function
def compute_metrics_by_groups(y: np.array, preds: np.array,
                              groups: np.array) -> pd.DataFrame:
    """Compute metrics by each group in groups.

    Args:
        y (np.array): Known labels, binarized.
        preds (np.array): Predicted labels, binarized.
        groups (np.array): Groups to be analyzed

    Returns:
        pd.DataFrame, where index are group values, columns are metric values.
        sMetrics are f1, precision, recall and support
    """

    def reduce_calc_metrics(idxs):
        idxs = np.array(idxs)
        out = compute_model_metrics(y=y[idxs], preds=preds[idxs])
        out.update({'support': len(idxs)})
        return out

    # get dict mapping grouup_value to index
    out = more_itertools.map_reduce(
        iterable=enumerate(groups),
        keyfunc=lambda x: x[1],  # group value
        valuefunc=lambda x: x[0],  # idx value
        reducefunc=reduce_calc_metrics  # convert to np.array
    )

    # Convert to pandas, index are group values, columns are metric values``
    out = pd.DataFrame(out).T.astype({
        'support': int
    }).round(2).sort_values('support', ascending=False)
    return out


# TODO: add tests for this function
def pretty_print_pandas(df: pd.DataFrame, index_rename: str = None) -> str:
    """Pretty print pandas dataframe contents, using tabulate module.

    Args:
        df (pd.DataFrame): input pandas dataframe
        index_rename (str, optional): how to rename index. Defaults to None
        (index is not renamed.)
    Returns:
        str: pretty printed table
    """

    headers = df.head(0).reset_index()
    if index_rename is not None:
        headers = headers.rename({'index': index_rename}, axis=1)
    headers = headers.columns.tolist()
    to_print = tabulate.tabulate(df.itertuples(),
                                 headers=headers,
                                 tablefmt='psql')
    return to_print
