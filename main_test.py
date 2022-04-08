"""Test for main app"""

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Import our app from main.py.
from main import WELCOME_MESSAGE, app
from starter.ml.data import process_data
from starter.ml.model import inference
from starter.train_model import CAT_FEATURES

# Instantiate the testing client with our app.
client = TestClient(app)


# Write tests using the same syntax as with the requests module.
def test_local_welcome_message():
    """Test if welcome message return code and contents as expected"""
    r = client.get('/')
    assert r.status_code == 200
    assert r.text == WELCOME_MESSAGE


def test_get_malformed():
    """Test if a malformed get query doest not return 200"""
    r = client.get("/anything")
    assert r.status_code != 200


@pytest.fixture(scope='session')
def offline_inference_artifacts():
    """Load data and offline predictions for testing"""
    model = joblib.load('model/model.joblib')
    encoder = joblib.load('model/encoder.joblib')
    lb = joblib.load('model/lb.joblib')
    offline_dataframe = pd.read_csv(
        'data_test/processed/census-fix-spaces-nodups.csv')
    X, _, _, _ = process_data(X=offline_dataframe.drop('salary', axis=1),
                              categorical_features=CAT_FEATURES,
                              label=None,
                              training=False,
                              encoder=encoder,
                              lb=lb)
    offline_predictions = inference(model=model, X=X)
    return {
        'offline_dataframe': offline_dataframe,
        'X': X,
        'offline_predictions': offline_predictions
    }


def test_prediction_match(offline_inference_artifacts):
    """Test if preprocessing and inference from file is equivalent to sending
    a request to the API.
    """
    # raw data as read from file
    raw_data = offline_inference_artifacts['offline_dataframe']
    parsed_data = raw_data.to_dict(orient='records')
    offline_predictions = offline_inference_artifacts[
        'offline_predictions'].ravel().tolist()
    for raw_json_input, expected_prediction in zip(parsed_data,
                                                   offline_predictions):
        r = client.post('/inference', json=raw_json_input)
        assert r.status_code == 200
        assert r.json() == {'prediction': expected_prediction}


def test_malformed_inference_request(offline_inference_artifacts):
    """Test if a malformed inference request doest not return code 200.
    """
    # raw data as read from file
    raw_data = offline_inference_artifacts['offline_dataframe'].iloc[0]
    r = client.post('/anything', json=raw_data.to_dict())
    assert r.status_code != 200
