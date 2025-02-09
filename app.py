from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Set the correct paths for models and templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# Define valid crops
VALID_CROPS = ["Rice", "Wheat", "Maize", "Barley"]

# Load models
try:
    yield_model = pickle.load(open(os.path.join(MODEL_DIR, "yield_model.pkl"), "rb"))
    price_model = pickle.load(open(os.path.join(MODEL_DIR, "price_model.pkl"), "rb"))
except FileNotFoundError as e:
    raise FileNotFoundError(f"Model file not found: {e}")

# Initialize LabelEncoder for crop labels
le = LabelEncoder()
le.fit(VALID_CROPS)

# Define routes
@app.route("/")
def index():
    return render_template("index.html")  # Ensure `index.html` is in the `templates` directory

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from the form
        data = request.form
        crop = data.get("crop")
        area = data.get("area")
        rainfall = data.get("rainfall")
        soil_quality = data.get("soil_quality")

        # Validate inputs
        if crop not in VALID_CROPS:
            return render_template("index.html", error=f"Invalid crop. Allowed values: {', '.join(VALID_CROPS)}")
        
        try:
            area = float(area)
            rainfall = float(rainfall)
            soil_quality = float(soil_quality)
        except ValueError:
            return render_template("index.html", error="Area, rainfall, and soil quality must be numeric values.")

        if area <= 0 or rainfall <= 0 or soil_quality < 0:
            return render_template("index.html", error="Ensure area, rainfall, and soil quality are positive numbers.")

        # Encode the crop label using LabelEncoder
        crop_encoded = le.transform([crop])[0]

        # Prepare data for yield prediction
        yield_features = pd.DataFrame([[crop_encoded, area, rainfall, soil_quality]],
                                       columns=["Crop", "Area", "Rainfall", "SoilQuality"])
        predicted_yield = yield_model.predict(yield_features)[0]

        # Prepare data for price prediction
        price_features = pd.DataFrame([[crop_encoded, predicted_yield]],
                                      columns=["Crop", "Yield"])
        predicted_price = price_model.predict(price_features)[0]

        # Return results to the same page
        return render_template("index.html", 
                       predicted_yield=round(predicted_yield, 2),
                       predicted_price=round(predicted_price, 2))


    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
