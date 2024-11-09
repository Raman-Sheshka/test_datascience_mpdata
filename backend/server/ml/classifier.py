import os
import pandas as pd
import joblib

# Path: backend/server/ml/classifier.py
# Here we are importing the trained LGBMClassifier from the save_models directory
# and using it to make predictions on the input data

class Classifier:
    def __init__(self):
        self.model = joblib.load(os.path.join(os.path.dirname(__file__), 'save_models', 'model.pkl'))
        self.encoder = joblib.load(os.path.join(os.path.dirname(__file__), 'save_models', 'scaler.pkl'))

    def preprocessing(self, input_data):
        # JSON to pandas DataFrame
        input_data = pd.DataFrame(input_data, index=[0])
        # fill missing values if necessary
        input_data.fillna(0.0, inplace=True)
        # scaling
        input_data = self.encoder.transform(input_data.values)
        return input_data

    def postprocessing(self, prediction):
        # do some postprocessing
        label = "Yes" if prediction == 1 else "No"
        return {"prediction": prediction, "label": label, "status": "OK"}

    def compute_predict(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.model.predict(input_data)[0]
            prediction  = self.postprocessing(prediction)
        except Exception as e:
            return {"status":"Error","message": str(e)}
        return prediction
