import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import cv2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

print("INFO: Getting Model File...")
subprocess.run(["python3", "model.py"], check=True)

app = Flask(__name__)
CORS(app)

model = load_model("models/bisindo_model.h5")

ALPHABET_MAPPING = [chr(i) for i in range(65, 91)]

def preprocess_image(img_file):
    """
    Preprocess the input image for the model
    
    :param img_file: File-like object of the input image
    :return: Preprocessed numpy array
    """
    
    img = Image.open(img_file).convert('RGB')
    img = img.resize((100, 100))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0 
    
    return img_array

@app.route('/', methods=['GET'])
def ping_server():
    return jsonify({
        'ping': 'Hello this is Genggam Makna AI!'
    })

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

    
def preprocess_video(video_path, target_fps=10):
    """
    Preprocess video by reducing its FPS and extracting frames.
    
    :param video_path: Path to the input video
    :param target_fps: Target FPS to process the video
    :return: List of frames
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / target_fps) if fps > target_fps else 1
    
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frames.append(frame)  # Save the frame for prediction
        
        frame_count += 1
    
    cap.release()
    return frames

def preprocess_frame(frame):
    """
    Preprocess the input video frame for the model
    
    :param frame: Numpy array of the video frame
    :return: Preprocessed numpy array
    """
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
    img = img.resize((100, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0 
    return img_array

def predict_frame(frame):
    """
    Predict the sign language for a single frame.
    
    :param frame: Numpy array of the frame
    :return: Predicted alphabet and confidence
    """
    processed_img = preprocess_frame(frame)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_alphabet = ALPHABET_MAPPING[predicted_class]
    confidence = float(np.max(prediction[0]))
    return predicted_alphabet, confidence


@app.route('/predict/video', methods=['POST'])
def predict_sign_language_video():
    """
    API endpoint for video sign language prediction with FPS adjustment.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    video_path = "/tmp/temp_video.mp4"
    file.save(video_path)

    try:
        frames = preprocess_video(video_path, target_fps=10)
        
        predictions = []
        confidences = []

        for frame in frames:
            predicted_alphabet, confidence = predict_frame(frame)
            predictions.append(predicted_alphabet)
            confidences.append(confidence)

        normalized_predictions = []
        current_alphabet = None
        current_count = 0

        for alphabet in predictions:
            if alphabet == current_alphabet:
                current_count += 1
            else:
                if current_count >= 7 and (not normalized_predictions or normalized_predictions[-1] != current_alphabet):
                    normalized_predictions.append(current_alphabet)
                current_alphabet = alphabet
                current_count = 1

        if current_count >= 7 and (not normalized_predictions or normalized_predictions[-1] != current_alphabet):
            normalized_predictions.append(current_alphabet)
        
        average_confidence = np.mean(confidences)

        return jsonify({
            'predicted_alphabet': normalized_predictions,
            'confidence': average_confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=4015)