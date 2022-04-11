# Deploying a Machine Learning Model on Heroku with FastAPI
## Introduction
This repository contains my submission for the course Deploying a Scalable ML Pipeline in Production, fourth course from Udacity `Machine Learning DevOps Engineer` NanoDegree. 
We train a classification model on [`Census Income Data Set`](https://archive.ics.uci.edu/ml/datasets/census+income) and deploy this on Heroku using FastAPI.

## How this repo is organized
Below we have the directory structure and annotations for relevant directoreis and files.
```bash
.
|-- Aptfile 
|-- LICENSE
|-- Makefile
|-- Procfile
|-- README.md
|-- README_starter.md
|-- data  # Data directory, almost everything tracked by DVC and stored on S3
|   |-- processed  # Procesed data
|   |-- raw  # raw data, as downloaded from starter repository
|   |   `-- census.csv.dvc
|   `-- train_test_data  # training and testing data, after preprocessing.
|-- data_test  # data used for tests, not tracked by DVC
|   `-- processed
|       `-- census-fix-spaces-nodups.csv
|-- dvc.lock
|-- dvc.yaml
|-- dvc_on_heroku_instructions.md
|-- main.py # main app functions
|-- model # model artifacts and metrics
|   `-- summary.json
|-- model_card.md
|-- EDA notebooks
|   `-- eda-1.ipynb
|-- reports # EDA reports using pandas profiling
|   |-- pandas-profiling-clean.html  # EDA on clean data
|   `-- pandas-profiling-raw.html  # EDA on raw data
|-- requirements.txt  # Python requirements
|-- run_live_post_get_live_api.py
|-- runtime.txt  # select Python version on Heroku
|-- sanitycheck.py
|-- screenshots # RUBRIC REQUIRED SCREENSHOTS
|   |-- continuous_deloyment.png # proof that continuous deployment is enabled on Heroku
|   |-- continuous_integration.png # proof that CI using github actions is set and passing
|   |-- dvcdag.png # DVC dag in ascii format
|   |-- dvcdag_dot_graphviz.png # DVC dag rendering dot format with graphviz
|   |-- dvcdag_outs.png # DVC dag (considering outputs) in ascii format
|   |-- dvcdag_outs_dot_graphviz.png # DVC dag (with outputs) rendering dot format with graphviz
|   |-- example.png # example of input body on FastAPI
|   |-- heroku_deploy_builds.png # image showing many builds on Heroku
|   `-- live_post_get.png # script, status codes and contents for both GET and POST METHODS
|-- setup.py # setup for package installation
|-- starter # our package
|   |-- __init__.py # random seed is set here
|   |-- ml
|   |   |-- __init__.py
|   |   |-- data.py # data preprocessing
|   |   |-- model.py # model training, inference, and metrics calculation
|   |   |-- test_data.py # tests for data.py
|   |   `-- test_model.py tests for model.py
|   `-- train_model.py # script for model training
|-- test_main.py # tests for main.py
|-- tmp_dvc_commands_run.sh # Draft commands used for runnning DVC pipeline (subsequent modifictaions were made directly on dvc.yaml file)
`-- tox.ini # configurations for flake8 and pytest 
```

## DVC Pipeline
DVC was used for entire data and modeling pipeline. Below are dvc.yaml file contents:
```yaml
stages:
  fix_spaces:
    cmd: sed 's/, /,/g' data/raw/census.csv > data/processed/census-fix-spaces.csv
    deps:
    - data/raw/census.csv
    outs:
    - data/processed/census-fix-spaces.csv
  remove_dups:
    cmd: awk '{counts[$0]++;if (counts[$0] == 1) {print $0}}' data/processed/census-fix-spaces.csv
      > data/processed/census-fix-spaces-nodups.csv
    deps:
    - data/processed/census-fix-spaces.csv
    outs:
    - data/processed/census-fix-spaces-nodups.csv
  train_eval_model:
    cmd: python3 starter/train_model.py
    deps:
    - data/processed/census-fix-spaces-nodups.csv
    - starter/train_model.py
    outs:
    - data/train_test_data/X_test.joblib
    - data/train_test_data/X_train.joblib
    - model/encoder.joblib
    - model/lb.joblib
    - model/model.joblib
    - model/slice_output.txt

    metrics:
    - model/summary.json:
        cache: false
    ```