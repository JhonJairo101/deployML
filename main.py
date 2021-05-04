import os
import joblib
from flask import Flask, jsonify, request
from app.modeler.modeler import Modeler

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "Método GET de Prueb con Flask"

@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.get_json()
    sepal_length = request_data['sepal_length']
    sepal_width = request_data['sepal_width']
    petal_length = request_data['petal_length']
    petal_width = request_data['petal_width']

    m = Modeler()
    m.fit()
    app.logger.debug('Modelo Entrenado...')
    prediction = m.predict([sepal_length, sepal_width, petal_length, petal_width])
    app.logger.debug('Prediction' + prediction)
    return jsonify({
           'Input' : {
               'sepal_length' : sepal_length,
               'sepal_width' : sepal_width,
               'petal_length' : petal_length,
               'petal_width' : petal_width,
           },
           'Prediction': prediction
      })


if __name__ == '__main__':
    app.run()

