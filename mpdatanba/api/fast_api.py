import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mpdatanba.ml_logic.ml_workflow import load_model, predict_model

app = FastAPI()

# pre-load the model
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
def predict(gp: float,
            fgm: float,
            fg_pca: float,
            oreb: float,
            reb: float,
            pts: float,
            ftm: float,
            ft_pca: float,
            tov: float
            ):
    data_test = np.array([0.52112676,
                          0.06060606,
                          0.28657315,
                          0.01886792,
                          0.04411765,
                          0.05090909,
                          0.02597403,
                          0.563,
                          0.13953488
                          ])


    model = app.state.model
    prediction = predict_model(model,
                               [data_test]
                               )
    print(prediction)
    print(type(prediction))
    return {'prediction': prediction.tolist()}

@app.get("/")
def root():
    return {'message': 'Hello Word!',
           }
