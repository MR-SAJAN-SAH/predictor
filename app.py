from flask import Flask, request, jsonify
import numpy as np
from joblib import load  # Updated import statement

# Load the model using joblib
model = load('mode.joblib')

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    cgpa = float(request.form.get('cgpa'))
    iq = float(request.form.get('iq'))
    profile_score = float(request.form.get('profile_score'))

    # Prepare input query as a numpy array
    input_query = np.array([[cgpa, iq, profile_score]])

    # Make prediction
    result = model.predict(input_query)[0]

    return jsonify({'placement': str(result)})

if __name__ == '__main__':
    app.run(debug=True)
