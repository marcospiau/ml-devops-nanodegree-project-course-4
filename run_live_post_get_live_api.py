"""Run live get and post methods on live API"""

import requests

APP_URL = 'https://mlops-fastapi-heroku-project.herokuapp.com'


def run_live_get():
    print(80 * '*')
    print('Running live GET')
    response = requests.get(APP_URL)
    print(f'response.status = {response.status_code}')
    print(f'response.content = {response.json()}')


def run_live_post():
    # Same data used as example on Pydantic BaseModel
    request_body = {
        'age': 39,
        'workclass': 'State-gov',
        'fnlgt': 77516,
        'education': 'Bachelors',
        'education-num': 13,
        'marital-status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 2174,
        'hours-per-week': 40,
        'native-country': 'United-States'
    }
    print(80 * '*')
    print('Running live POST')
    response = requests.post(f'{APP_URL}/inference/', json=request_body)
    print(f'response.status = {response.status_code}')
    print(f'response.content = {response.json()}')


if __name__ == '__main__':
    run_live_get()
    run_live_post()
