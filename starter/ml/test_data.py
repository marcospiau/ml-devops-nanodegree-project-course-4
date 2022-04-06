import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.utils.estimator_checks import check_is_fitted

from .data import process_data

# Create fake data
cat_cols = ['cat_1', 'cat_2']
num_cols = ['num_1', 'num_1']
target_col = 'target'
df = pd.DataFrame({
    'target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    'quant_1': [0.1, 0.2, 0.3, 0.9, 10, 12, 13, 18, 9, 10],
    'quant_2': [0, 1, 2, 3, 5.5, 6.6, 7.7, 8.8, 9.9, 10],
    'cat_1': ['0', '1', '2', '0', '1', '2', '0', '1', '2', '0'],
    'cat_2':
    ['Good', 'Bad', 'Bad', 'Bad', 'Bad', 'Bad', 'Bad', 'Bad', 'Good', 'Bad']
})
# repeat data ten times
df = pd.concat([df] * 10, ignore_index=True)

# get transformed data and fitted estimators
X_train, y_train, encoder, lb = process_data(X=df,
                                             categorical_features=cat_cols,
                                             label=target_col,
                                             training=True)

# Simulate behavior in test (or inference) mode
X_test, y_test, _, _ = process_data(X=df,
                                    categorical_features=cat_cols,
                                    label=target_col,
                                    training=False,
                                    encoder=encoder,
                                    lb=lb)


@pytest.mark.parametrize(['obj', 'expected_type'], [(lb, LabelBinarizer),
                                                    (encoder, OneHotEncoder)],
                         ids=['one_hot_encoder', 'label_binarizer'])
def test_correct_type(obj, expected_type):
    """Test encoders types are as expected"""
    assert isinstance(obj, expected_type)


@pytest.mark.parametrize('obj', [encoder, lb],
                         ids=['one_hot_encoder', 'label_binarizer'])
def test_encoders_are_fitted(obj):
    """Test if encoders are fitted"""
    check_is_fitted(obj)


@pytest.mark.parametrize(['actual', 'expected'], [(X_test, X_train),
                                                  (y_test, y_train)],
                         ids=['X', 'y'])
def test_process_data_inference(actual, expected):
    """Test if process_data function works on inference (and test) mode"""
    assert np.allclose(actual, expected), 'Values should be equal'


def test_target_cardinality():
    """Test if cardinality of target data is correct"""
    assert len(set(df[target_col])) == len(set(y_train))


def test_features_cardinality():
    """Check if features cardinality are correct"""
    # One output column for each numeric column
    num_cols_len = len(num_cols)
    # One output column for each distinct value of categorical column
    cat_cols_cardinality = sum(len(set(df[col])) for col in cat_cols)
    assert X_train.shape[1] == num_cols_len + cat_cols_cardinality
