from dotenv import load_dotenv
import os
import requests
from tqdm import tqdm  
import tensorflow as tf
from tensorflow.keras.models import load_model

load_dotenv()
model_url = os.getenv("GCLOUD_BUCKET_MODEL_URL")

local_model_path = "models/bisindo_model.h5"

if not os.path.exists(local_model_path):
    print("Downloading model...")
    response = requests.get(model_url, stream=True)  
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  

    with open(local_model_path, "wb") as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress:
        for data in response.iter_content(block_size):
            f.write(data)
            progress.update(len(data))
    print("Model downloaded successfully.")


print("Loading the model...")
model = load_model(local_model_path)
print("Model loaded successfully.")
