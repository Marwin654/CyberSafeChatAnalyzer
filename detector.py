import joblib

# Load trained model
model = joblib.load("models/spam_detector.pkl")

# Prediction function
def predict_message(message):
    prediction = model.predict([message])[0]
    return prediction


# Test prediction
sample = "HEy bro we meeting today?"

result = predict_message(sample)

print("Prediction:", result)