from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model & encoder
model = joblib.load("model/sales_forecast.pkl")
encoder = joblib.load("model/encoder.pkl")
feature_order = joblib.load("model/feature_order.pkl")

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Sales Forecasting API! Use /predict to get predictions."})

@app.route('/predict', methods=['POST'])
def predict_sales():
    try:
        # Get JSON request data
        data = request.get_json()
        df = pd.DataFrame([data])

        # Convert 'Date' and extract date-related features
        df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d", errors='coerce')
        if df['Date'].isna().any():
            return jsonify({"error": "Invalid date format"}), 400

        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df.drop(columns=['Date'], inplace=True)

        # Encode categorical features
        categorical_cols = ['Store_Type', 'Location_Type', 'Region_Code', 'Discount']
        encoded_features = encoder.transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

        # Drop original categorical columns and merge encoded features
        df.drop(columns=categorical_cols, inplace=True)
        df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        # **Ensure feature order matches training**
        df = df[feature_order]

        # Make prediction
        prediction = model.predict(df)
        return jsonify({"predicted_sales": round(float(prediction[0]), 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)