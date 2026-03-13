import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.preprocessing import image_dataset_from_directory

# -------------------------------------------------------------------
# Configuration Parameters
# -------------------------------------------------------------------
DATASET_DIR = "dataset"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "crop_disease_model.h5")
CLASS_INDICES_PATH = "class_indices.json"

BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
EPOCHS = 20

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

def train_model():
    # -------------------------------------------------------------------
    # 1. Data Loading and Preprocessing
    # -------------------------------------------------------------------
    print("Loading dataset...")
    # Using image_dataset_from_directory for easy structured loading (Train split)
    train_dataset = image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    # Validation split
    validation_dataset = image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_dataset.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    if num_classes == 0:
        print("ERROR: No classes found. Please place your images inside the 'dataset' directory organized by class folders.")
        return

    # Cache and prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # -------------------------------------------------------------------
    # 2. CNN Architecture
    # -------------------------------------------------------------------
    # The architecture consists of 3 Conv2D+MaxPooling blocks to extract spatial features,
    # followed by a Dense layer with Dropout for classification.
    model = Sequential([
        # Data Preprocessing/Rescaling (normalize pixel values between 0 and 1)
        Rescaling(1./255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        
        # Block 1: Extracts low-level features (edges, colors)
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Block 2: Extracts mid-level features (textures, specific patterns)
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Block 3: Extracts high-level features (complex abstract shapes)
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Classifier output
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), # Regularization to prevent overfitting
        Dense(num_classes, activation='softmax') # Output layer probabilities
    ])

    # -------------------------------------------------------------------
    # 3. Model Compilation
    # -------------------------------------------------------------------
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    model.summary()

    # -------------------------------------------------------------------
    # 4. Model Training
    # -------------------------------------------------------------------
    print(f"Starting training for {EPOCHS} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS
    )

    # -------------------------------------------------------------------
    # 5. Save the trained model
    # -------------------------------------------------------------------
    model.save(MODEL_PATH)
    print(f"Model successfully saved to {MODEL_PATH}")

    # We comment out the class_indices.json generation to preserve the detailed agronomic solutions
    # mapping = {}
    # for idx, name in enumerate(class_names):
    #     mapping[str(idx)] = {
    #         "plant": name.split('_')[0] if '_' in name else "Unknown",
    #         "disease": name.replace('_', ' '),
    #         "solution": "Consult an agricultural expert for the best treatment plan."
    #     }
    # 
    # with open(CLASS_INDICES_PATH, "w") as f:
    #     json.dump(mapping, f, indent=4)
    # print(f"Class mapping successfully saved to {CLASS_INDICES_PATH}")

if __name__ == "__main__":
    train_model()
