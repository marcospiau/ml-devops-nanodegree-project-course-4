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
|-- README.md # This file
|-- README_starter.md # Starter README.md
|-- data  # data folder
|   |-- processed
|   |   |-- census-fix-spaces-nodups.csv
|   |   `-- census-fix-spaces.csv
|   |-- raw ## 
|   |   |-- census.csv
|   |   `-- census.csv.dvc
|   `-- train_test_data
|       |-- X_test.joblib
|       `-- X_train.joblib
|-- data_test
|   `-- processed
|       `-- census-fix-spaces-nodups.csv
|-- dvc.lock
|-- dvc.yaml
|-- dvc_on_heroku_instructions.md
|-- main.py
|-- model
|   |-- encoder.joblib
|   |-- lb.joblib
|   |-- model.joblib
|   |-- slice_output.txt
|   `-- summary.json
|-- model_card_template.md
|-- notebooks
|   `-- eda-1.ipynb
|-- rascunhos_mudancas.txt
|-- reports
|   |-- pandas-profiling-clean.html
|   `-- pandas-profiling-raw.html
|-- requirements.txt
|-- run_live_post_get_live_api.py
|-- runtime.txt
|-- sanitycheck.py
|-- screenshots
|   |-- continuous_deloyment.png
|   |-- continuous_integration.png
|   |-- dvcdag.png
|   |-- dvcdag_dot_graphviz.png
|   |-- dvcdag_outs.png
|   |-- dvcdag_outs_dot_graphviz.png
|   |-- example.png
|   |-- heroku_deploy_builds.png
|   `-- live_post_get.png
|-- setup.py
|-- starter
|   |-- __init__.py
|   |-- ml
|   |   |-- __init__.py
|   |   |-- data.py
|   |   |-- model.py
|   |   |-- test_data.py
|   |   `-- test_model.py
|   `-- train_model.py
|-- starter.egg-info
|   |-- PKG-INFO
|   |-- SOURCES.txt
|   |-- dependency_links.txt
|   `-- top_level.txt
|-- test_main.py
|-- tmp_dvc_commands_run.sh
`-- tox.ini
```

