from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

nn_pipe = load('../models/nn_pipeline.joblib')

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'Predict Beer Type'

def format_features(genre: str,	age: int, income: int, spending: int):
    return {
        'review_aroma': [review_aroma],
        'review_appearance': [review_appearance],
        'review_palate': [review_palate],
        'review_tastee': [review_taste],
        'beer_abv': [beer_abv]
    }

@app.get("/beers/type")
def predict(genre: str,	age: int, income: int, spending: int):
    features = format_features(review_aroma,	review_appearance, review_palate, review_taste, beer_abv)
    obs = pd.DataFrame(features)
    pred = nn_pipe.predict(obs)
    return JSONResponse(pred.tolist())