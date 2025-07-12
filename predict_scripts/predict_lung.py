import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.activations import swish
import numpy as np
from PIL import Image

# Define custom FixedDropout layer
class FixedDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

# Add both FixedDropout and swish activation to custom_objects
custom_objects = {
    'FixedDropout': FixedDropout,
    'swish': swish
}

# Set model path
MODEL_PATH = 'final_model_finetuned1.h5'

# Load model
model = load_model(MODEL_PATH, custom_objects=custom_objects)

# Define class names (update as per your model's class labels)
class_names = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']

# Ask user for image path
image_path = input("Enter the path to the image: ")

# Load and preprocess image
img = Image.open(image_path).convert('RGB')
img = img.resize((224, 224))  # Adjust based on your model's input
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]

print(f"Predicted class: {predicted_class}")
