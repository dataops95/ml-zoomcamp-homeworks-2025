import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

app=Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_features = np.array(data['features'].reshape (1,-1))
    scaled_features = scaler.transform(input_features)
    prediction = model.predict(scaled_features)
    return jsonify({'predicted_price': prediction [0]})

if __name__=="__main__":
    app.run(debug=True, port=5001)