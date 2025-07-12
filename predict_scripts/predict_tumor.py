import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os

# Load the custom layer class (needed if your model uses a custom layer)
class GetItem(tf.keras.layers.Layer):
    def __init__(self, index, **kwargs):
        super(GetItem, self).__init__(**kwargs)
        self.index = index

    def call(self, inputs):
        return inputs[self.index]

    def get_config(self):
        config = super(GetItem, self).get_config()
        config.update({"index": self.index})
        return config

# Register the custom layer globally
from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({'GetItem': GetItem})

# Step 1: Preprocess the input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # Match the input size to the model
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image (assuming your model was trained with normalized data)
    return img_array

# Step 2: Load the trained model
model = load_model('models/tumor_detection_enhanced_model.h5', custom_objects={'GetItem': GetItem})

# Step 3: Predict whether the image has a tumor or not
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

def predict_tumor(img_path):
    # Preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))  # Load the image
    img_array = img_to_array(img)  # Convert PIL Image to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)

    # Get the class with the maximum probability
    predicted_class = np.argmax(prediction, axis=1)

    # Check the predicted class and return the result
    if (predicted_class[0] == 0).all():  # No tumor detected
        result = "No tumor detected"
    else:
        result = "Tumor detected"

    # Convert the PIL image to a NumPy array
    

    return result

    

# This function will save and process the uploaded image
def handle_upload(upload_file):
    # Save the uploaded image with a secure filename
    filename = secure_filename(upload_file.filename)
    upload_path = os.path.join('uploads', filename)  # Ensure this folder exists
    upload_file.save(upload_path)
    
    # Process the uploaded image for prediction
    result = predict(upload_path)
    return result
