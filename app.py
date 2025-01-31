from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

# Load the trained model
model = tf.keras.models.load_model('trained_model.h5')

# Initialize Flask app
app = Flask(__name__)

# Define the class names (change this based on your model's class names)
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 
                'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
                'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 
                'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
                'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
                'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Define the image preprocessing function
def prepare_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the file temporarily
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)

        # Prepare the image and make predictions
        img_array = prepare_image(img_path)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_names[predicted_class]

        # Return the prediction as a JSON response
        return jsonify({'prediction': predicted_label})

@app.route('/')
def home():
    return 'Plant Disease Prediction API is running!'

@app.route('/favicon.ico')
def favicon():
    return '', 204  # Returns a no-content response for favicon requests

if __name__ == '__main__':
    # Create uploads folder if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
        
    # Start the Flask app
    app.run(debug=True)