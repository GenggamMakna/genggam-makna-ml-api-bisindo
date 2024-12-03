import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess

print("INFO: Getting Model File...")
subprocess.run(["python3", "model.py"], check=True)

app = Flask(__name__)
CORS(app)

model = load_model("models/sibid_model.h5")

ALPHABET_MAPPING = [chr(i) for i in range(65, 91) if chr(i) not in ["J", "Z"]]

def preprocess_image(img_file):
    """
    Preprocess the input image for the model
    
    :param img_file: File-like object of the input image
    :return: Preprocessed numpy array
    """
    
    img = Image.open(img_file).convert("RGB")
    
    img = img.resize((200, 200))
    
    img_array = image.img_to_array(img)
    
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/predict/image', methods=['POST'])
def predict_sign_language():
    """
    API endpoint for sign language prediction
    """
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        processed_img = preprocess_image(file)
        
        prediction = model.predict(processed_img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        predicted_alphabet = ALPHABET_MAPPING[predicted_class]
        
        confidence = float(np.max(prediction[0]))
        
        return jsonify({
            'predicted_alphabet': predicted_alphabet,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8014)