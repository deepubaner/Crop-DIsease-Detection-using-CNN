import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

# Optional: Disable TF oneDNN warnings if they clutter console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Configuration
MODEL_PATH = os.path.join("model", "crop_disease_model.h5")
CLASS_INDICES_PATH = "class_indices.json"
UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained CNN model
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"WARNING: Model not found at {MODEL_PATH}. Prediction endpoint will fail until model is trained.")
except Exception as e:
    print(f"Error loading model: {e}")

# Load the class indices mapping
class_indices = {}
try:
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, "r") as f:
            class_indices = json.load(f)
        print("Class mapping loaded successfully.")
except Exception as e:
    print(f"Error loading class indices: {e}")

# Helper function to preprocess the image for the model
def prepare_image(img_path):
    # Load image with standard PlantVillage 256x256 resolution
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    # The model handles rescaling (1./255) internally via the Rescaling layer
    # So we just expand dimensions to match batch format: shape (1, 256, 256, 3)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return img_array_expanded

@app.route("/", methods=["GET"])
def index():
    """Render the main frontend UI."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint to accept an image upload, run the model, and return results."""
    if model is None:
        return jsonify({"error": "Model not trained. Please run 'python train_model.py' first."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading."}), 400

    if file:
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Prepare image and predict
            processed_image = prepare_image(file_path)
            predictions = model.predict(processed_image)
            
            # Get the highest probability index
            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            confidence = float(np.max(predictions))

            # Retrieve info from JSON map
            str_idx = str(predicted_class_idx)
            if str_idx in class_indices:
                result = class_indices[str_idx]
                plant_name = result.get("plant", "Unknown Plant")
                disease_name = result.get("disease", "Unknown Disease")
                solution = result.get("solution", "No solution mapped.")
            else:
                plant_name = "Unknown"
                disease_name = "Class index not found in mapping"
                solution = "N/A"

            # Clean up the uploaded file to save space (optional, comment out to keep history)
            if os.path.exists(file_path):
                os.remove(file_path)

            return jsonify({
                "success": True,
                "confidence": round(confidence * 100, 2),
                "plant": plant_name,
                "disease": disease_name,
                "solution": solution
            })

        except Exception as e:
            # Clean up file on error
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
