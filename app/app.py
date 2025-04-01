from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "ML Model is Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "features" not in data:
        return jsonify({"error": "'features' key is missing from the request"}),400

    features = data["features"]

    for feature_set in features:
        if len(feature_set) != 4:
            return jsonify({"error": "Each feature set must have exactly 4 values"}),400
        if not all(isinstance(x, (float, int)) for x in feature_set):
            return jsonify({"error": "Each feature value must be a float or int"}),400

    features_array = np.array(features)
    predictions = model.predict(features_array)

    response = {
        "predictions": predictions.tolist()
    }

    return jsonify(response)

@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "ok"
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000) #check your port number ( if it is in use, change the port number)
