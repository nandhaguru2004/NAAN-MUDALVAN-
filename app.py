
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess data
def load_and_preprocess_data():
    df = pd.read_csv('customer.csv')

    # Drop rows with missing values (or handle as needed)
    df.dropna(inplace=True)

    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Label encode binary columns
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        if X[col].nunique() == 2:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

    # One-hot encode categorical variables with >2 categories
    X = pd.get_dummies(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, y, scaler, label_encoders, X.columns

# Train model
def train_model(X, y):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model

# Prepare data and model at startup
X_orig, X_scaled, y, scaler, label_encoders, feature_columns = load_and_preprocess_data()
model = train_model(X_scaled, y)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Apply label encoding
        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])

        # Handle missing columns by adding them
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # One-hot encode input to match training data structure
        input_df = pd.get_dummies(input_df)

        # Align input with training columns
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = float(model.predict_proba(input_scaled)[0][1])

        return jsonify({
            'churn_prediction': int(prediction),
            'churn_probability': probability
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
