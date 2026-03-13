import os
import json
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling

# -------------------------------------------------------------------
# Configuration Parameters
# -------------------------------------------------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "crop_disease_model.h5")
CLASS_INDICES_PATH = "class_indices.json"

BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
EPOCHS = 5 # Reduced for quicker demonstration training

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

def preprocess(features):
    # tfds provides dicts, we pull out image and label
    image = features['image']
    label = features['label']
    # Resize the image to expected size
    image = tf.image.resize(image, IMAGE_SIZE)
    return image, label

def train_model():
    print("Loading PlantVillage dataset via TensorFlow Datasets...")
    
    # Load dataset. 'plant_village' has no predefined test/val splits, only 'train'.
    # We will split it manually 80/20 for train/val.
    dataset, info = tfds.load(
        'plant_village', 
        with_info=True, 
        as_supervised=False # We use False first to show features dict processing
    )
    
    # Actually, as_supervised=True returns (image, label) directly, which is simpler!
    dataset_supervised = tfds.load(
        'plant_village', 
        split='train',
        as_supervised=True
    )
    
    # Total examples: 54,303.
    # Split 80% train, 20% validation
    total_samples = info.splits['train'].num_examples
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    
    # Shuffle and split
    dataset_supervised = dataset_supervised.shuffle(10000, seed=123)
    train_dataset = dataset_supervised.take(train_size)
    validation_dataset = dataset_supervised.skip(train_size)

    # Preprocess (resize images)
    def resize_img(image, label):
        return tf.image.resize(image, IMAGE_SIZE), label

    train_dataset = train_dataset.map(resize_img, num_parallel_calls=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.map(resize_img, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    class_names = info.features['label'].names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes.")

    # -------------------------------------------------------------------
    # 2. CNN Architecture
    # -------------------------------------------------------------------
    model = Sequential([
        Rescaling(1./255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
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

    # Generate mapping for frontend
    mapping = {}
    for idx, name in enumerate(class_names):
        # class names are like "Apple___Cedar_apple_rust"
        parts = name.split("___")
        plant = parts[0].replace("_", " ") if len(parts) > 0 else "Unknown"
        disease = parts[1].replace("_", " ") if len(parts) > 1 else name
        mapping[str(idx)] = {
            "plant": plant,
            "disease": disease,
            "solution": "Consult a local agricultural extension for proper fungicides or treatments based on severity." if disease.lower() != "healthy" else "Maintain proper care."
        }
    
    with open(CLASS_INDICES_PATH, "w") as f:
        json.dump(mapping, f, indent=4)
    print(f"Class mapping successfully saved to {CLASS_INDICES_PATH}")

if __name__ == "__main__":
    train_model()
