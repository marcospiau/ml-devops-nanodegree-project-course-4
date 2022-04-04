# Script to train machine learning model.

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)-8s: %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def joblib_dump_and_log(obj, path: str, description: str) -> None:
    """Dump file using joblib and print logging message with location

    Args:
        obj (object): object to dump
        path (str): where to save the file
        description (str): object description
    """
    joblib.dump(obj, path)
    logging.info('Saving %s to %s', description, path)


def json_dump_and_log(obj, path: str, description: str) -> None:
    """Dump file using json and print logging message with location

    Args:
        obj (object): object to dump
        path (str): where to save the file
        description (str): object description
    """
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)
    logging.info('Saving %s to %s', description, path)


# Add the necessary imports for the starter code.

# Setup directories
MODEL_DIR = Path('model')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path('data')
(DATA_DIR / 'train_test_data').mkdir(parents=True, exist_ok=True)

# Add code to load in the data.
logging.info('Reading data')
data = pd.read_csv('data/processed/census-fix-spaces-nodups.csv')
data.info(show_counts=True)

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
logging.info('Running train/test split')
train, test = train_test_split(data, test_size=0.20, stratify=data['salary'])

cat_features = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
]

logging.info('Running data preprocessing')
X_train, y_train, encoder, lb = process_data(train,
                                             categorical_features=cat_features,
                                             label='salary',
                                             training=True)
joblib_dump_and_log(X_train, DATA_DIR / 'train_test_data/X_train.joblib',
                    'X_train')

# Persist encoder and lb to disk, will be used for inferene
joblib_dump_and_log(encoder, MODEL_DIR / 'encoder.joblib',
                    'Categorical encoder')
joblib_dump_and_log(lb, MODEL_DIR / 'lb.joblib', 'Label binarizer')

# Process the test data with the process_data function.

X_test, y_test, _, _ = process_data(X=data,
                                    categorical_features=cat_features,
                                    label='salary',
                                    training=False,
                                    encoder=encoder,
                                    lb=lb)

joblib_dump_and_log(X_test, DATA_DIR / 'train_test_data/X_test.joblib',
                    'X_test')

# Train and save a model.
logging.info('Training model')
model = train_model(X_train, y_train)

logging.info('Saving model to %s', MODEL_DIR / 'model.joblib')
joblib_dump_and_log(model, MODEL_DIR / 'model.joblib', 'model')

# Calculate and save metrics to json
train_metrics = compute_model_metrics(y=y_train,
                                      preds=inference(model, X_train))
test_metrics = compute_model_metrics(y=y_test, preds=inference(model, X_test))
logging.info('%s:\n%s', 'train_metrics', train_metrics)
logging.info('%s:\n%s', 'test_metrics', test_metrics)

json_dump_and_log({
    'train': train_metrics,
    'test': test_metrics
}, MODEL_DIR / 'summary.json', 'Train and test metrics')
