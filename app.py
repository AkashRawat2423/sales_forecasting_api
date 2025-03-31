from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback

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
        print("Received Request for Prediction")
        data = request.get_json()
        df = pd.DataFrame([data])

        # Ensure 'Date' is processed correctly
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format="%Y-%m-%d")

        # Extract time-based features
        df['Year'] = df['Date'].dt.year
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x in [5, 6] else 0)
        df.drop(columns=['Date'], inplace=True)

        # One-Hot Encode Categorical Features (Ensuring Correct Columns)
        categorical_cols = ['Store_Type', 'Location_Type', 'Region_Code', 'Discount']

        for col in categorical_cols:
            if col not in df.columns:
                df[col] = "Unknown"  # Handle missing categories gracefully

        expected_categories = encoder.feature_names_in_

        # Ensure all categorical columns exist in the correct order
        df = df.reindex(columns=expected_categories, fill_value="Unknown")

        # Encode the categorical variables
        encoded_features = encoder.transform(df[expected_categories])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(expected_categories))

        df = df.drop(columns=categorical_cols).reset_index(drop=True)
        df = pd.concat([df, encoded_df], axis=1)

        # Ensure all expected columns exist, fill missing ones with 0
        for col in feature_order:
            if col not in df.columns:
                df[col] = 0  # Fill missing feature columns with 0

        df = df[feature_order]  # Ensure correct column order

        print("Processed Input Data:\n", df.head())

        # Make prediction
        prediction = model.predict(df)
        print("Prediction:", prediction[0])

        return jsonify({"predicted_sales": round(prediction[0], 2)})

    except Exception as e:
        print(f" ERROR in Prediction: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)