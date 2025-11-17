from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    values = [
        float(request.form["mileage"]),
        float(request.form["engine_temp"]),
        float(request.form["rpm"]),
        float(request.form["oil_pressure"]),
        float(request.form["fuel_efficiency"]),
    ]

    final_features = scaler.transform([values])
    prediction = model.predict(final_features)[0]

    result = "⚠ Maintenance Required Soon" if prediction == 1 else "✅ Vehicle is in Good Condition"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)
