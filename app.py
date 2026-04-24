import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Configuration
TFLITE_MODEL_PATH = os.path.join("model", "crop_disease_model.tflite")
CLASS_INDICES_PATH = "class_indices.json"
UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load TFLite model
interpreter = None
input_details = None
output_details = None

try:
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
    except ImportError:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite model loaded successfully.")
except Exception as e:
    print(f"Error loading TFLite model: {e}")

# Load class indices
class_indices = {}
try:
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    print("Class mapping loaded successfully.")
except Exception as e:
    print(f"Error loading class indices: {e}")


def prepare_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)  # rescaling is inside the model
    return np.expand_dims(img_array, axis=0)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if interpreter is None:
        return jsonify({"error": "Model not loaded."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file in request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        img = prepare_image(file_path)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        predicted_class_idx = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))

        str_idx = str(predicted_class_idx)
        if str_idx in class_indices:
            result = class_indices[str_idx]
            plant_name = result.get("plant", "Unknown Plant")
            disease_name = result.get("disease", "Unknown Disease")
            solution = result.get("solution", "No solution mapped.")
        else:
            plant_name = "Unknown"
            disease_name = "Class index not found"
            solution = "N/A"

        return jsonify({
            "success": True,
            "confidence": round(confidence * 100, 2),
            "plant": plant_name,
            "disease": disease_name,
            "solution": solution
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
