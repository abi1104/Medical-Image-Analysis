import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load the trained model
model = load_model("models/chest_xray_model_optimized.h5")

# Class labels (update if necessary based on training order)
class_labels = ['NORMAL', 'PNEUMONIA']

# Function to preprocess the uploaded image and make predictions
def predict_pneumonia(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model input size
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    # Return the prediction result as a string
    return f"Predicted class: {class_labels[predicted_class]} ({confidence * 100:.2f}% confidence)"

