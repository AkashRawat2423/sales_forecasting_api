from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback
import numpy as np

app = Flask(__name__)

# Load trained model & transformers
model = joblib.load("model/sales_forecast.pkl")  # Trained Model
encoder = joblib.load("model/encoder.pkl")  # One-Hot Encoder
scaler, scaling_columns = joblib.load("model/scaler.pkl")  # Scaler & Columns
feature_order = joblib.load("model/feature_order.pkl")  # Feature Order

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Sales Forecasting API! Use /predict to get predictions."})

@app.route('/predict', methods=['POST'])
def predict_sales():
    try:
        print("📥 Received Prediction Request")

        # Validate JSON Request
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Empty request body. Please send valid JSON."}), 400
        except Exception as e:
            return jsonify({"error": f"Invalid JSON format: {str(e)}"}), 400

        # Convert JSON to DataFrame
        df = pd.DataFrame([data])

        # Convert 'Date' to datetime and extract features
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format="%Y-%m-%d")
        df['Year'] = df['Date'].dt.year
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x in [5, 6] else 0)
        df.drop(columns=['Date'], inplace=True)

        # Categorical Features
        categorical_cols = ['Store_Type', 'Location_Type', 'Region_Code', 'Discount']

        # Ensure all categorical columns are present
        for col in categorical_cols:
            if col not in df.columns:
                df[col] = "Unknown"

        # Ensure categorical feature order is correct
        cat_df = pd.DataFrame()
        for col in encoder.feature_names_in_:
            cat_df[col] = df[col] if col in df.columns else "Unknown"

        # One-Hot Encoding
        encoded_features = encoder.transform(cat_df)
        encoded_df = pd.DataFrame(
            encoded_features,
            columns=encoder.get_feature_names_out(encoder.feature_names_in_)
        )

        # Drop original categorical columns and merge encoded features
        df.drop(columns=categorical_cols, inplace=True)
        df = pd.concat([df, encoded_df], axis=1)

        # Handle missing lag-based features
        lag_features = ['Sales_Lag_7', 'Sales_Moving_Avg_7', 'Sales_Lag_30']
        for col in lag_features:
            if col not in df.columns:
                df[col] = 0  # Assign 0 if missing (adjust based on training strategy)

        # Apply log transformation on lag-based features before scaling
        for col in lag_features:
            df[col] = np.log1p(df[col])

        # Ensure correct feature order
        prediction_df = pd.DataFrame(0, index=range(1), columns=feature_order)
        for col in df.columns:
            if col in prediction_df.columns:
                prediction_df[col] = df[col]

        print("🔍 Final Prediction DataFrame Shape:", prediction_df.shape)
        print("📝 Final Prediction Columns:", prediction_df.columns.tolist())

        # Scaling numeric features
        scaling_cols_to_use = [col for col in scaling_columns if col in feature_order]
        print(f"📊 Scaling Columns: {scaling_cols_to_use}")

        if scaling_cols_to_use:
            to_scale = prediction_df[scaling_cols_to_use].copy()

            # Ensure correct column order for scaling
            scale_cols_df = pd.DataFrame()
            for col in scaling_columns:
                scale_cols_df[col] = to_scale[col] if col in to_scale.columns else 0

            # Apply scaling
            scaled_values = scaler.transform(scale_cols_df)

            # Update only the scaled columns
            for i, col in enumerate(scaling_columns):
                if col in scaling_cols_to_use:
                    prediction_df[col] = scaled_values[:, i]

        # Make Prediction
        prediction = model.predict(prediction_df)

        # Apply log transformation fix (if model was trained with log sales)
        predicted_sales = max(0, np.expm1(prediction[0]))

        print(f"📊 Predicted Sales: {predicted_sales}")

        return jsonify({"predicted_sales": round(predicted_sales, 2)})

    except Exception as e:
        print(f"❌ ERROR in Prediction: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)