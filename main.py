import json
import logging
import os
from collections import OrderedDict
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from starter.ml.data import process_data
from starter.ml.model import inference
from starter.train_model import CAT_FEATURES

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)-8s: %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
WELCOME_MESSAGE = """\
Welcome to my first model API!
We use a simple decision tree to predict higher salaries.
Feel free to reach me out at name@domain.com
"""

if 'DYNO' in os.environ and os.path.isdir('.dvc'):
    os.system('dvc config core.no_scm true')
    if os.system('dvc pull') != 0:
        exit('dvc pull failed')
    os.system('rm -r .dvc .apt/usr/lib/dvc')

app = FastAPI()


class Input(BaseModel):
    """Data structure used for model input"""
    age: int
    workclass: Literal['?', 'Federal-gov', 'Local-gov', 'Never-worked',
                       'Private', 'Self-emp-inc', 'Self-emp-not-inc',
                       'State-gov', 'Without-pay']
    fnlgt: int
    education: Literal['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th',
                       '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors',
                       'Doctorate', 'HS-grad', 'Masters', 'Preschool',
                       'Prof-school', 'Some-college']
    education_num: int = Field(alias='education-num')
    marital_status: Literal['Divorced', 'Married-AF-spouse',
                            'Married-civ-spouse', 'Married-spouse-absent',
                            'Never-married', 'Separated',
                            'Widowed'] = Field(alias='marital-status')
    occupation: Literal['?', 'Adm-clerical', 'Armed-Forces', 'Craft-repair',
                        'Exec-managerial', 'Farming-fishing',
                        'Handlers-cleaners', 'Machine-op-inspct',
                        'Other-service', 'Priv-house-serv', 'Prof-specialty',
                        'Protective-serv', 'Sales', 'Tech-support',
                        'Transport-moving']
    relationship: Literal['Husband', 'Not-in-family', 'Other-relative',
                          'Own-child', 'Unmarried', 'Wife']
    race: Literal['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other',
                  'White']
    sex: Literal['Female', 'Male']
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: Literal['?', 'Cambodia', 'Canada', 'China', 'Columbia',
                            'Cuba', 'Dominican-Republic', 'Ecuador',
                            'El-Salvador', 'England', 'France', 'Germany',
                            'Greece', 'Guatemala', 'Haiti',
                            'Holand-Netherlands', 'Honduras', 'Hong',
                            'Hungary', 'India', 'Iran', 'Ireland', 'Italy',
                            'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua',
                            'Outlying-US(Guam-USVI-etc)', 'Peru',
                            'Philippines', 'Poland', 'Portugal', 'Puerto-Rico',
                            'Scotland', 'South', 'Taiwan', 'Thailand',
                            'Trinadad&Tobago', 'United-States', 'Vietnam',
                            'Yugoslavia'] = Field(alias='native-country')

    # salary: str
    # Make possible request using both attribute name and field alias
    class Config:
        allow_population_by_field_name = True

    def to_pandas(self):
        # column names and order must be kept same as seen on training
        map_columns = OrderedDict([
            ('age', 'age'),
            ('workclass', 'workclass'),
            ('fnlgt', 'fnlgt'),
            ('education', 'education'),
            ('education_num', 'education-num'),
            ('marital_status', 'marital-status'),
            ('occupation', 'occupation'),
            ('relationship', 'relationship'),
            ('race', 'race'),
            ('sex', 'sex'),
            ('capital_gain', 'capital-gain'),
            ('capital_loss', 'capital-loss'),
            ('hours_per_week', 'hours-per-week'),
            ('native_country', 'native-country'),
        ])
        df = pd.DataFrame([self.dict()]).rename(map_columns,
                                                axis=1)[map_columns.values()]
        return df


@app.post('/inference', response_class=PlainTextResponse)
async def make_predictions(input_body: Input):
    logging.info('Receiving prediction')

    logging.info('Receiving model artifacts')
    model = joblib.load('model/model.joblib')
    encoder = joblib.load('model/encoder.joblib')
    lb = joblib.load('model/lb.joblib')
    df = input_body.to_pandas()

    logging.info('Processing data')
    X, _, _, _ = process_data(X=df,
                              categorical_features=CAT_FEATURES,
                              label=None,
                              training=False,
                              encoder=encoder,
                              lb=lb)

    logging.info('Making prediction')
    prediction = inference(model=model, X=X).item()
    return json.dumps({'prediction': prediction})


@app.get('/', response_class=PlainTextResponse)
def welcome_message():
    return WELCOME_MESSAGE
