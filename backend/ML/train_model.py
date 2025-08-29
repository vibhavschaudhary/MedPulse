import pandas as pd
import sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv(r'S:\Projects\MedPulse\Backend\ML\dataSets\synthetic_triage_data.csv')
class DataProcessor:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def drop_missing(self):
        """Drop rows with missing values"""
        self.df = self.df.dropna()
        return self   # return self so you can chain methods

    def get_df(self):
        """Return the processed dataframe"""
        return self.df

processor = DataProcessor(df)

df_proc = processor.drop_missing().get_df()

df_proc= df_proc.dropna()
df.dropna(subset=['triage'], inplace=True)
# Add this after loading the data
print("--- Triage Class Distribution ---")
print(df['triage'].value_counts())
features = [
    'age', 'gender', 'chest pain type', 'blood pressure', 
    'cholesterol', 'max heart rate', 'exercise angina', 
    'plasma glucose', 'hypertension', 'heart_disease'
]
target = 'triage'
X = df[features]
X.fillna(X.mean(), inplace=True)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
joblib.dump(model, 'triage_model.pkl')
print("\nModel training complete and saved as 'triage_model.pkl'.")
print(model)
results = pd.DataFrame({
    "Actual Duration": y_test.values,
    "Predicted Duration": predictions
})

print("\n--- Sample Predictions ---")
print(results.head(50))
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")