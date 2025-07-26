from flask import Flask, render_template, render_template_string, request, jsonify
import pickle
import os
import json
from datetime import datetime
from extract_features import extract_features # Ensure this imports the UPDATED extract_features
from flask import send_from_directory
from flask_cors import CORS
import numpy as np

# Your existing bot-like heuristic functions (keep these for now, but be aware of them)
def is_bot_like_keyboard(keyboard_features_segment): # This function will now receive the first 3 features
    # Assuming keyboard_features_segment is [avg_key_hold, avg_interkey_latency, typing_duration]
    key_hold, interkey_latency, duration = keyboard_features_segment
    return key_hold < 30 or interkey_latency < 50 or duration < 1000

def is_bot_like_mouse(mouse_features_segment): # This function will now receive features 4,5,6
    # Assuming mouse_features_segment is [avg_mouse_speed, max_mouse_speed, avg_mouse_accel]
    speed, max_speed, accel = mouse_features_segment
    return speed < 0.1 or max_speed < 0.2 or accel < 0.05

app = Flask(__name__)
CORS(app)

# ✅ Load the model (ensure this is the new model.pkl trained on 14 features)
try:
    with open("model.pkl", "rb") as f: # Use the new model name
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: trained_behavioral_model.pkl not found. Please ensure it's in the same directory.")
    # Exit or handle gracefully if model is not found
    model = None # Or raise an error to prevent further execution

@app.route("/")
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return render_template_string(f.read())
    
@app.route("/demo.html")
def demo():
    with open("demo.html", "r", encoding="utf-8") as f:
        return render_template_string(f.read())

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/x-icon')

@app.route("/access-denied")
def access_denied():
    # Render the HTML template for the access denied page
    return render_template("access_denied.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"result": "Error: Model not loaded."}), 500

    try:
        data = request.get_json()

        # Extract ALL 14 numerical features as a single list
        # This must match the order and number of features your model was trained on!
        all_features = extract_features(data)

        print(f"--- Features received for prediction: {all_features}") # ADD THIS LINE
        print(f"--- Data type of all_features: {type(all_features)}, Length: {len(all_features)}") # AND THIS LINE

        # Ensure the input to the model is a 2D array (even for single prediction)
        model_input = np.array(all_features).reshape(1, -1) # Reshape for single sample prediction

        # Get model prediction
        predicted_label = model.predict(model_input)[0] # 0 for genuine, 1 for fraudulent

        # Get probability of fraud (optional, but good for confidence)
        # fraud_probability = model.predict_proba(model_input)[0][1]


        # --- Heuristic Bot Checks (Refined to use segments from all_features) ---
        # These segments correspond to the order in all_features
        avg_key_hold, avg_interkey_latency, typing_duration = all_features[0], all_features[1], all_features[2]
        avg_mouse_speed, max_mouse_speed, avg_mouse_accel = all_features[3], all_features[4], all_features[5]
        paste_detected = all_features[8] # Directly from all_features list

        keyboard_bot = is_bot_like_keyboard([avg_key_hold, avg_interkey_latency, typing_duration])
        mouse_bot = is_bot_like_mouse([avg_mouse_speed, max_mouse_speed, avg_mouse_accel])

        # Final decision based on the RandomForest model's prediction
        # Temporarily comment out the heuristic override to test the RF model's direct prediction
        # if (keyboard_bot or mouse_bot) or paste_detected:
        #     final_result = "FRAUDULENT USER (Heuristic Detected)"
        # elif predicted_label == 1:
        #     final_result = "FRAUDULENT USER (ML Model Detected)"
        # else:
        #     final_result = "✅ Genuine user. Login success."


        # Simplified decision: Rely primarily on the ML model's prediction
        if predicted_label == 1:
            final_result = "FRAUDULENT USER (ML Model Detected)"
        else:
            final_result = "✅ Genuine user. Login success."

        # If you want to layer heuristics, you can do this:
        # if (keyboard_bot or mouse_bot) or paste_detected:
        #     final_result = "FRAUDULENT USER (Heuristic Detected)"
        # elif predicted_label == 1:
        #     final_result = "FRAUDULENT USER (ML Model Detected)"
        # else:
        #     final_result = "✅ Genuine user. Login success."


        return jsonify({"result": final_result})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"result": f"Error during prediction: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)