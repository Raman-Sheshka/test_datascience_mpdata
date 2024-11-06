import glob
import os
import time
import pandas as pd
import lightgbm as lgb
from mpdatanba.utils import create_folder_if_not_exists
from mpdatanba import PARENT_BASE_PATH

def preprocess_data():
    # Preprocess the data
    print("Preprocessing data...")
    time.sleep(2)
    print("Data preprocessed!")
    return None

def train_model():
    # Train the model
    print("Training model...")
    time.sleep(2)
    print("Model trained!")
    return None

def evaluate_model():
    # Evaluate the model
    print("Evaluating model...")
    time.sleep(2)
    print("Model evaluated!")
    return None

def predict():
    # Make predictions
    print("Making predictions...")
    time.sleep(2)
    print("Predictions made!")
    return None

def predict_model(model,
                  X,
                  threshold:float=0.5,
                  ) :
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(model, lgb.basic.Booster):
        proba_pred = model.predict(X)
        y_pred = (proba_pred > threshold).astype(int)
    else:
        # use sklearn model
        y_pred = model.predict(X)
    return y_pred

def load_model(model_target:str = 'local',
               sklearn:bool=False,
               ):
    """Load the latest model from disk
    Args:
        model_target (str, optional): config params. Defaults to 'local'.
        sklearn (bool, optional): config params. Defaults to False.

    Returns:
        : latest saved model
    """
    # Load the model
    print("Loading model...")
    if model_target == "local":
        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(PARENT_BASE_PATH,
                                             "save_models"
                                             )
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
        if not sklearn:
            latest_model = lgb.Booster(model_file=most_recent_model_path_on_disk)

        print("Model loaded from local disk")

        return latest_model

def save_model(model,
               model_target:str = 'local'):
    # Save the model
    print("Saving model...")
    if model_target == "local":
        local_model_directory = os.path.join(PARENT_BASE_PATH,
                                             "save_models"
                                             )
        create_folder_if_not_exists(local_model_directory)
        model_file = os.path.join(local_model_directory,
                          "model_selected.txt"
                          )
        model.save_model(model_file)
    print("Model saved!")
    return None
