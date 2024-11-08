# app.py
from flask import Flask
from flask_restful import Api, Resource, reqparse

APP = Flask(__name__)
API = Api(APP)

class Predict(Resource):

    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument('petal_length')
        parser.add_argument('petal_width')
        parser.add_argument('sepal_length')
        parser.add_argument('sepal_width')

        args = parser.parse_args()  # creates dict

        out = {'Prediction': [0]}

        return out, 200


API.add_resource(Predict, '/predict')

if __name__ == '__main__':
    APP.run(debug=True,
            port='8000'
            )
