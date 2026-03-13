# 🌿 AgroVision AI - Crop Disease Detection System

AgroVision AI is an end-to-end deep learning web application designed to empower farmers and agricultural experts. It detects crop diseases from plant leaf images with high accuracy using a Convolutional Neural Network (CNN) and provides actionable agronomic solutions.

## ✨ Key Features

- **Advanced Deep Learning Analysis**: Utilizes a custom-trained CNN built with TensorFlow/Keras tailored for agricultural imagery.
- **Interactive Modern Interface**: A premium "Glassmorphism" UI with smooth micro-animations, designed for usability and responsiveness across all devices.
- **Multi-Language Support**: Built-in translation for English, Hindi, and Punjabi, ensuring accessibility for local farmers.
- **Comprehensive Disease Mapping**: Identifies 38 different plant disease classes across major crops (Wheat, Tomato, Potato, Corn, Chilli) and provides specific pesticide/treatment solutions.
- **Real-Time Visual Feedback**: Drag-and-drop image upload, instant scanning animations, and dynamic confidence badges.

## 🛠️ Technology Stack

**Backend Model & Server:**
- **Python 3.8+**
- **TensorFlow / Keras**: CNN Architecture, Model Training & Inference
- **Flask**: Web Framework and REST-like API (`/predict`)
- **NumPy**: Image array preprocessing
- **Werkzeug**: Secure file uploading

**Frontend UI/UX:**
- **HTML5 & CSS3 (Vanilla)**: For layout and advanced styling (Glassmorphism, CSS Variables, Flexbox/Grid, Animations)
- **Vanilla JavaScript**: For asynchronous requests (Fetch API), UI state management, and file handling
- **Google Translate API**: For localized language switching
- **Lucide Icons**: For clean vector iconography

## 📂 Project Structure

```text
crop/
│
├── app.py                      # Main Flask application and API endpoints
├── train_model.py              # Script to build, train, and save the CNN model
├── train_model_tfds.py         # Alternative training script using TensorFlow Datasets
├── create_sample_dataset.py    # Utility to generate synthetic dataset for quick testing
├── download_dataset.py         # Scripts for downloading datasets locally
├── class_indices.json          # JSON mapping of CNN output indices to plant/disease/solutions
├── requirements.txt            # Python dependencies list
│
├── dataset/                    # Directory for storing raw training data
├── model/                      
│   └── crop_disease_model.h5   # The compiled and trained Deep Learning model
│
├── static/
│   ├── style.css               # Main stylesheet (Glassmorphism, Dark Mode variables)
│   ├── script.js               # Frontend application logic (Drag/Drop, Fetch API)
│   └── uploads/                # Temporary directory for user-uploaded images
│
└── templates/
    └── index.html              # Main web interface HTML document
```

## 🚀 Setup & Installation Guide

### 1. Prerequisites
Ensure you have **Python 3.8 or higher** installed on your system.
Verify your Python installation:
```bash
python --version
```

### 2. Install Dependencies
Navigate to the project folder and install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Model Training (Optional if model already exists)
If `crop_disease_model.h5` does not exist in the `model/` folder, you need to train the model first.

Ensure you have your dataset in the `dataset/` folder, organized by class folders. You can also generate a quick sample dataset for testing:
```bash
python create_sample_dataset.py
```
Train the model:
```bash
python train_model.py
```
*Note: The script automatically handles image augmentation, resizing (256x256), and saves the trained `.h5` file.*

### 4. Running the Application
Start the Flask web server:
```bash
python app.py
```

### 5. Accessing the Platform
Open your web browser and navigate to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

## 📖 How To Use

1. **Upload an Image**: Click on the upload zone or drag and drop an image of an affected crop leaf.
2. **Analyze**: Click the "**Analyze Leaf**" button. The frontend sends an async request to the Flask server.
3. **View Diagnosis**: The model processes the image and returns:
   - **Confidence Score**: How certain the AI is of its prediction.
   - **Plant Name & Disease**: The exact health status of the leaf.
   - **Treatment & Solution**: Actionable advice and chemical (pesticide/fungicide) recommendations.
4. **Translate**: Use the top-left language switcher to view the results and platform in Hindi or Punjabi.

## 🧠 Neural Network Architecture

The custom CNN operates effectively on `256x256x3` RGB images.
1. **Rescaling Layer**: Normalizes pixel values (`1./255`) for efficient learning.
2. **Convolutional Blocks**: Successive `Conv2D` layers (32, 64, 128 filters) equipped with `ReLU` activation, each followed by `MaxPooling2D` for feature extraction and down-sampling spatial dimensions.
3. **Flatten & Dense**: The extracted 2D feature maps are flattened and passed through a highly dense fully-connected network with `Dropout` layers to prevent overfitting.
4. **Output Layer**: A final `.Dense` layer using `softmax` activation that outputs probability distribution corresponding to the classes available in `class_indices.json`.
