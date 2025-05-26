from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("model/symptom_model.pkl")

@app.route("/")
def home():
    return "🩺 Symptom Checker API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        symptoms_text = data.get("symptoms", "")

        if not symptoms_text.strip():
            return jsonify({"error": "Symptoms input is empty."}), 400

        prediction = model.predict([symptoms_text])[0]
        probabilities = model.predict_proba([symptoms_text])[0]
        confidence = max(probabilities)

        result = {
            "predicted_disease": prediction,
            "confidence": round(float(confidence), 3)
        }

        # Add uncertainty warning
        if confidence < 0.6:
            result["warning"] = "⚠️ Low confidence — please consult a doctor."

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run app
if __name__ == "__main__":
    app.run(debug=True)
