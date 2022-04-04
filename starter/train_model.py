# Script to train machine learning model.

import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv('../data/processed/census-fix-spaces-nodups.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label='salary', training=True
)
# Proces the test data with the process_data function.

X_test, y_test, _, _ = process_data(
    X=data,
    categorical_features=cat_features,
    label='salary',
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save a model.
