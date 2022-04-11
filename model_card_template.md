# Model Card
## Model Details
This model uses census data to predict if one has a income above of below $50K.
We train a decition tree classifier  ([`sklearn.tree.DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)) with max depth of 6 and a random seed of 42, and default values for the remaining hyperparameters. this project focus was not to obtain the best possible metrics, so we did not perform hyperparameter search and this is left as future work. This model was trained by Marcos Piau Vieira on April 2022.

## Intended Use
This model is trained for pratice and educational purposes and should be used for that. Its API can be used to request predictions given input data.


## Training Data
Training data is the [**Census Income DataSet**](https://archive.ics.uci.edu/ml/datasets/census+income), freely available in UCI machine learning repository. Following the instructions, this dataset was downloaded from starter code repository and can be downloaded from this link [this link](https://github.com/udacity/nd0821-c3-starter-code/blob/master/starter/data/census.csv).
Raw data file contained additional spaces and duplicated rows, and we used linux commands `sed` and `awk`, respectively, to solve these problems. We use [`sklearn.preprocessing.OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) to encode categorical features in a one hot fashion and [`sklearn.preprocessing.LabelBinarizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html) to transform the target column to binary values; numeric features are left as is, since the model we use is robust to unscaled features.


## Evaluation Data

For evaluation data, we apply stratified random sampling to total data, with 20% for evaluation and the remaining for training, keeping `salary` target variable evenly distributed across train and validation partitions. The full code for this step is in `starter/train_model.py`, below is the code excerpt focusing on this step:
```python
    train, test = train_test_split(data,
                                   test_size=0.20,
                                   stratify=data['salary'],
                                   random_state=RANDOM_SEED)
```

## Metrics
We use precision, f1 and recall as evaluation metrics. Table below shows these metrics for both train and test.

| Path               | test.f1   | test.precision   | test.recall   | train.f1   | train.precision   | train.recall   |
|--------------------|-----------|------------------|---------------|------------|-------------------|----------------|
| model/summary.json | 0.6556    | 0.805            | 0.5529        | 0.6365     | 0.7891            | 0.5334         |


## Ethical Considerations
This model uses features that can induce racial and sexual bias, that probably should not be used to take decisions (eg. credit approval). There is also skewness in the distribution of some variables like sex and country of origin, that can bias the model toward particular values of these informations.

## Caveats and Recommendations
Some modifications can be done to improve this model:
 - a thoroughly investigation of biases involved in all features to improve fairness
 - exploration of more techniques and hyperparameters could led to better metrics
 - more tests for the deployed API, more specifically to ensure requests limits are within required for real life usage scenarios
 - a more detailed EDA can be used to clean the data better or add more filters to train and test data. For example, for rows with `workclass` feature having values in set {`Never-worked`, `Without-pay`} we have perfect scores and only one example for each value of this feature; a more careful EDA process probably would remove these examples from training and test data, since people that does not have a job will neve have a higher income value

## Additional information
More information about model cards can be found in paper [Model Cards for Model Reporting
](https://arxiv.org/abs/1810.0399).