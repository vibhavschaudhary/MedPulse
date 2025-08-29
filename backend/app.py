from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# --- Load the Trained Model ---
# Make sure 'triage_model.pkl' is in the same folder as this script
try:
    model = joblib.load('S:\Projects\MedPulse\Backend\ML\Triage_model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model file not found. Make sure 'triage_model.pkl' is in the directory.")
    model = None

# --- Define the Features (must match the model training) ---
features = [
    'age', 'gender', 'chest pain type', 'blood pressure',
    'cholesterol', 'max heart rate', 'exercise angina',
    'plasma glucose', 'hypertension', 'heart_disease'
]

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    # Get the JSON data from the request
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid JSON input'}), 400

    # Convert the JSON data into a pandas DataFrame
    try:
        patient_df = pd.DataFrame([data], columns=features)
    except Exception as e:
        return jsonify({'error': f'Error creating DataFrame: {e}'}), 400

    # Make a prediction
    try:
        prediction = model.predict(patient_df)
    except Exception as e:
        return jsonify({'error': f'Error making prediction: {e}'}), 500

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    # Runs the Flask app on your local machine
    app.run(debug=True, port=5000)