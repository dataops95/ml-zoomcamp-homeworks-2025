import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


app=Flask(__name__)
## Load the model
model=pickle.load(open('model.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    """
    API Endpoint that accepts JSON input, applies the pre-loaded scaler,
    and returns the model prediction.
    """
    try:
        # 1. Robust JSON extraction
        if request.is_json:
            json_data = request.json
        else:
            json_data = request.get_json(force=True)
        
        data = json_data.get('data')
        
        # 2. Validation
        if not data:
            return jsonify({"error": "Missing 'data' field in payload"}), 400

        # 3. Pre-processing
        # We need to ensure the input shape matches what the scaler expects (1, -1).
        # Assuming 'data' is a dictionary { "feature1": val, "feature2": val ... }
        # We convert values to a list, then to a numpy array, then reshape.
        input_data = np.array(list(data.values())).reshape(1, -1)
        
        # 4. Scaling (The missing step)
        # We must use the exact same scaler object used during training.
        scaled_data = scaler.transform(input_data)
        
        # 5. Prediction
        output = model.predict(scaled_data)
        
        # 6. Formatting response
        # We use float() to ensure it is JSON serializable (numpy types are not)
        return jsonify({
            "prediction": float(output[0])
        })
        
    except ValueError as ve:
        # specific handling for shape mismatches or type errors
        return jsonify({"error": f"Value Error: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))


if __name__=="__main__":
    app.run(debug=True)
