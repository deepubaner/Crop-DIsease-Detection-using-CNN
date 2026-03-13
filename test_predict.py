import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import glob

# Load model
model = tf.keras.models.load_model(os.path.join("model", "crop_disease_model.h5"))

# Load indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Find potato image
potato_images = glob.glob(r"dataset\Potato*/*.jpg")
if not potato_images:
    print("No potato image found!")
    exit(1)

test_img_path = potato_images[0]
print(f"Testing on {test_img_path}")

img = image.load_img(test_img_path, target_size=(256, 256))
img_array = image.img_to_array(img)
img_array_expanded = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array_expanded)
predicted_class_idx = np.argmax(predictions, axis=1)[0]
confidence = float(np.max(predictions))

str_idx = str(predicted_class_idx)
result = class_indices.get(str_idx, {})
print(f"Prediction: {result.get('plant')} {result.get('disease')} with {confidence*100:.2f}% confidence.")
