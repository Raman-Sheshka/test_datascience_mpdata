import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mpdatanba import PARENT_BASE_PATH
from mpdatanba.ml_logic.ml_workflow import MLWorkflow
from pydantic import BaseModel

app = FastAPI(debug=True)

# pre-load the model
mlworkflow = MLWorkflow(model_type='sklearn')
mlworkflow.load_model()
mlworkflow.encoder = mlworkflow.load_encoder()

app.state.model = mlworkflow.model


class InputFeatures(BaseModel):
    gp: float
    min: float
    pts: float
    fgm: float
    fga: float
    fg_pca: float
    three_p_made: float
    three_pa: float
    three_p_pca: float
    ftm: float
    fta: float
    ft_pca: float
    oreb: float
    dreb: float
    reb: float
    ast: float
    stl: float
    blk: float
    tov: float

# for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# model info endpoint
@app.get("/info")
# use async def if you are using async functionsn no need here
def model_info():
    model = app.state.model
    model_name = model.__class__.__name__
    model_params = model.get_params()
    model_information = {'model info':{
        'model name': model_name,
        'model parameters': model_params
        }
    }
    return model_information

# prediction endpoint
@app.post("/predict")
async def predict(input_features: InputFeatures):
    data_test = np.array([[input_features.gp,
                          input_features.min,
                          input_features.pts,
                          input_features.fgm,
                          input_features.fga,
                          input_features.fg_pca,
                          input_features.three_p_made,
                          input_features.three_pa,
                          input_features.three_p_pca,
                          input_features.ftm,
                          input_features.fta,
                          input_features.ft_pca,
                          input_features.oreb,
                          input_features.dreb,
                          input_features.reb,
                          input_features.ast,
                          input_features.stl,
                          input_features.blk,
                          input_features.tov
                          ]])
    prediction = mlworkflow.compute_predict(X=data_test)
    print(f"return prediction : {prediction}")
    return {'prediction': prediction.tolist()}

# root endpoint
@app.get("/")
def root():
    return {'message': 'Welcome to the NBA player prediction API',
           }

# health check endpoint
@app.get("/health")
def check_health():
    return {"status": "ok"}
