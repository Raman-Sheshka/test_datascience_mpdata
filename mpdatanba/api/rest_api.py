from flask import Flask, render_template, request, redirect
from mpdatanba.ml_logic.ml_workflow import load_model, predict_model
from flask_restful import Resource, reqparse, Api
#from flask_cors import CORS
import os
import numpy as np
#import prediction

app = Flask(__name__)
# pre-load the model
model = load_model()

#cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

# class Test(Resource):
#     def get(self):
#         return 'Welcome to, Test App API!'

#     def post(self):
#         try:
#             value = request.get_json()
#             if(value):
#                 return {'Post Values': value}, 201

#             return {"error":"Invalid format."}

#         except Exception as error:
#             return {'error': error}

class Predict(Resource):
    # def get(self):
    #     return {"error":"Invalid Method."}

    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument('gp')
        parser.add_argument('fgm')
        parser.add_argument('fg_pca')
        parser.add_argument('oreb')
        parser.add_argument('reb')
        parser.add_argument('pts')
        parser.add_argument('ftm')
        parser.add_argument('ft_pca')
        parser.add_argument('tov')
        args = parser.parse_args() # creates dict

        #data_test = np.fromiter(args.values(), dtype=float) # convert input to array
        #print(data_test)
        #prediction = predict_model(model, [data_test])

        #output = {'prediction': prediction.tolist()}
        output = {'prediction': [0]}
        return output, 200
        try:
            data = request.get_json()
            predict = prediction.predict_mpg(data)
            predictOutput = predict
            return {'predict':predictOutput}

        except Exception as error:
            return {'error': error}

# api.add_resource(Test,'/')
# api.add_resource(GetPredictionOutput,'/getPredictionOutput')
api.add_resource(Predict, '/predict')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True,
            host='0.0.0.0',
            port=port,
            )
