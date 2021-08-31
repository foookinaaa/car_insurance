import os
import dill
from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
import numpy as np
import pandas as pd

MODEL_FILE = os.path.join(os.path.dirname(__file__), 'data_output/model.bin')
ENCODER_FILE = os.path.join(os.path.dirname(__file__), 'data_output/encoder.bin')
META_MODEL_FILE = os.path.join(os.path.dirname(__file__), 'data_output/meta_model.bin')
MODEL_SHAP_FILE = os.path.join(os.path.dirname(__file__), 'data_output/booster.bin')
SERVER_PORT = int(os.environ.get('PORT', '8080'))

app = Flask('APP')

with open(MODEL_FILE, 'rb') as f:
    predictor = dill.load(f)  # pickle like

with open(ENCODER_FILE, 'rb') as f:
    encoder = dill.load(f)  # pickle like

with open(META_MODEL_FILE, 'rb') as f:
    meta_model = dill.load(f)  # pickle like

with open(MODEL_SHAP_FILE, 'rb') as f:
    booster = dill.load(f)  # pickle like



def predict(x):
    p, fe = predictor(pd.DataFrame([x]), booster)
    return int(p), fe


@app.route('/predict', methods=['POST'])
def predict_handler():
    request_data = request.json
    for col in [
 'AGE',
 'GENDER',
 'RACE',
 'DRIVING_EXPERIENCE',
 'EDUCATION',
 'INCOME',
 'CREDIT_SCORE',
 'VEHICLE_OWNERSHIP',
 'VEHICLE_YEAR',
 'MARRIED',
 'CHILDREN',
 'POSTAL_CODE',
 'ANNUAL_MILEAGE',
 'VEHICLE_TYPE',
 'SPEEDING_VIOLATIONS',
 'DUIS',
 'PAST_ACCIDENTS']:
        assert col in request_data.keys()

    return jsonify((predict(request_data)))



@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=SERVER_PORT)  # single thread server must be replaced with gunicorn

