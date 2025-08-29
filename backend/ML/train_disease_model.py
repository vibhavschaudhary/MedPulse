from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# --- Load all models and helper files from the 'ML' subfolder ---
try:
    # Triage model assets
    triage_model = joblib.load('triage_model.pkl')
    
    # Disease model assets
    disease_model = joblib.load('disease_model.pkl')
    disease_label_encoder = joblib.load('disease_label_encoder.pkl')
    all_symptoms = joblib.load('all_symptoms.pkl')

    # Symptom severity mapping
    df_severity = pd.read_csv(r'ML\Symptom-severity.csv')
    df_severity['Symptom'] = df_severity['Symptom'].str.strip().str.replace(' ', '_')
    symptom_to_severity = dict(zip(df_severity['Symptom'], df_severity['weight']))
    
    print("✅ All models loaded successfully.")

except FileNotFoundError as e:
    print(f"❌ ERROR: A required file was not found in the 'ML' folder. Please run the training scripts.")
    print(f"Details: {e}")
    # We exit here because the app is not functional without the models
    exit()

# --- Features for the triage model ---
features_triage = [
    'age', 'gender', 'chest pain type', 'blood pressure', 'cholesterol', 
    'max heart rate', 'exercise angina', 'plasma glucose', 'hypertension', 'heart_disease'
]

# --- Single Endpoint for Combined Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON input'}), 400

    try:
        # --- 1. Triage Severity Prediction ---
        patient_df = pd.DataFrame([data], columns=features_triage)
        severity_pred = triage_model.predict(patient_df)[0]

        # --- 2. Disease Prediction ---
        symptoms = data.get("symptoms", []) # Safely get the symptoms list
        
        # Create a numerical vector based on the provided symptoms, matching the training script
        input_vector = pd.DataFrame(0, index=[0], columns=all_symptoms)
        top_symptoms_list = []
        for sym in symptoms:
            cleaned_sym = sym.strip().replace(" ", "_")
            if cleaned_sym in all_symptoms:
                severity = symptom_to_severity.get(cleaned_sym, 0)
                input_vector[cleaned_sym] = severity
                top_symptoms_list.append((cleaned_sym, severity))

        # Predict the disease
        disease_pred_encoded = disease_model.predict(input_vector)[0]
        disease_name = disease_label_encoder.inverse_transform([disease_pred_encoded])[0]

        # --- 3. Explanation (top symptoms) ---
        # This logic is preserved exactly as you designed it
        explained = sorted(top_symptoms_list, key=lambda x: x[1], reverse=True)[:3]

        return jsonify({
            'severity_level': severity_pred,
            'likely_disease': disease_name,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("🚀 Starting Flask server for combined prediction... test on port 5000")
    app.run(host='0.0.0.0', port=5000)

