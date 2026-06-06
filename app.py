from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load models
spam_model = joblib.load("models/spam_detector.pkl")
tox_model = joblib.load("models/toxicity_detector.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    spam_prediction = None
    spam_confidence = None

    tox_prediction = None
    tox_confidence = None

    risk = None

    if request.method == "POST":

        message = request.form["message"]

        # Spam Detection
        spam_prediction = spam_model.predict([message])[0]

        spam_confidence = round(
            max(spam_model.predict_proba([message])[0]) * 100,
            2
        )

        # Toxicity Detection
        tox_prediction = tox_model.predict([message])[0]

        tox_confidence = round(
            max(tox_model.predict_proba([message])[0]) * 100,
            2
        )

        # Risk Analysis
        risk = "Low Risk"

        if spam_prediction == "spam":
            risk = "High Risk"

        if tox_prediction == 1:
            risk = "High Risk"

    return render_template(
        "index.html",
        spam_prediction=spam_prediction,
        spam_confidence=spam_confidence,
        tox_prediction=tox_prediction,
        tox_confidence=tox_confidence,
        risk=risk
    )


if __name__ == "__main__":
    app.run(debug=True)