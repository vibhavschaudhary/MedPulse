from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load(r'triage_model.pkl')
    # Load the severity mapping needed for the disease model
    df_severity = pd.read_csv('ML/Symptom-severity.csv')
    df_severity['Symptom'] = df_severity['Symptom'].str.strip().str.replace(' ', '_')
    symptom_to_severity = dict(zip(df_severity['Symptom'], df_severity['weight']))
    print("Model loaded successfully.")
except FileNotFoundError:
    print("ERROR: triage_model.pkl not found. Please run train_model.py first.")
    exit()

features = [
    'age', 'gender', 'chest pain type', 'blood pressure', 'cholesterol', 
    'max heart rate', 'exercise angina', 'plasma glucose', 'hypertension', 'heart_disease'
]

@app.route('/predict_triage', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        patient_df = pd.DataFrame([data], columns=features)
        prediction = model.predict(patient_df)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Starting Flask server... You can now test with Postman.")
    app.run(debug=False, port=5000)