import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mpdatanba.ml_logic.ml_workflow import load_model

app = FastAPI()
app.state.model = load_model()

# for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict/")
def predict():
    data = pd.DataFrame({
        'PTS': [110],
        'FGM': [40],
        'FGA': [90],
        'FG%': [0.44],
        '3PM': [10],
        '3PA': [30],
        '3P%': [0.33],
        'FTM': [20],
        'FTA': [30],
        'FT%': [0.67],
        'OREB': [10],
        'DREB': [30],
        'REB': [40],
        'AST': [20],
        'TOV': [15],
        'STL': [10],
        'BLK': [5],
        'PF': [20],
        'FP': [100],
        'DD2': [5],
        'TD3': [1]
    })

    model = app.state.model
    #prediction = model.predict(data)
    #return {'prediction': prediction.tolist()}
    return dict(prediction=np.array([1, 0]))

@app.get("/")
def root():
    return {'message': 'Hello Word!',
           }
