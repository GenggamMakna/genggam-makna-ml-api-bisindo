# ML Service: Sign Language Recognition API

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/) [![Flask](https://img.shields.io/badge/Flask-2.3.3-green)](https://flask.palletsprojects.com/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange)](https://www.tensorflow.org/) [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A RESTful API for recognizing sign language from images and videos using a deep learning model. The API supports real-time and batch predictions with advanced preprocessing and confidence scoring.

---

## Features

- **Sign Language Prediction**: Recognizes hand signs from images and videos.
- **Alphabet Mapping**: Maps predictions to 24 alphabets (excluding `J` and `Z`).
- **Video Support**: Processes videos with customizable FPS adjustments for accurate predictions.
- **Model Management**: Automatically downloads the model if not present locally.

---

## Tech Stack

- **Backend**: [Flask](https://flask.palletsprojects.com/)  
- **Deep Learning Framework**: [TensorFlow](https://www.tensorflow.org/)  
- **Data Handling**: [Pillow](https://python-pillow.org/) and [OpenCV](https://opencv.org/)  
- **Deployment**: Flask with CORS enabled for cross-origin requests  

---

## Endpoints

### Ping Server

- **URL**: `/`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "ping": "Hello this is Genggam Makna AI!"
  }
  ```

### Predict Sign Language (Image)

- **URL**: `/predict/image`
- **Method**: `POST`
- **Request**:  
  `Content-Type: multipart/form-data`  
  Upload an image file using the `file` key.

- **Response**:
  ```json
  {
    "predicted_alphabet": "A",
    "confidence": 0.98
  }
  ```

### Predict Sign Language (Video)

- **URL**: `/predict/video`
- **Method**: `POST`
- **Request**:  
  `Content-Type: multipart/form-data`  
  Upload a video file using the `file` key.

- **Response**:
  ```json
  {
    "predicted_alphabet": ["A", "B", "C"],
    "confidence": 0.92
  }
  ```

---

## How It Works

1. **Image Preprocessing**: 
   - Resizes input images to `224x224`.
   - Normalizes pixel values to `[0, 1]`.

2. **Video Processing**: 
   - Extracts frames at the target FPS (default: `10`).
   - Predicts each frame and aggregates results to ensure stable predictions.

3. **Model Download**:
   - Downloads the trained TensorFlow model from a remote location (URL set in `.env`).

---

## Setup

### Prerequisites

- Python 3.12 or later
- Required Python packages (install via `requirements.txt`).

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/GenggamMakna/genggam-makna-ml-api
   cd ml-service
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file and specify the model URL:
   ```env
   GCLOUD_BUCKET_MODEL_URL=<your_model_url>
   ```

4. Run the application:
   ```bash
   python3 app.py
   ```

---

## Directory Structure

```plaintext
.
├── app.py            # Main application file
├── model.py          # Handles model downloading and loading
├── models/           # Directory for storing the model file
├── requirements.txt  # Python dependencies
└── .env              # Environment variables
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For further questions or contributions, please contact:
- **Name**: Rama Diaz
- **Website**: [xann.my.id](https://xann.my.id)
- **Email**: ramadiaz221@gmail.com