import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model ONCE during cold start (outside the function)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.h5')
model = tf.keras.models.load_model(MODEL_PATH)

def prepare_image(image_file, image_size=(128, 128)):
    # Open from the uploaded file stream
    original = Image.open(image_file).convert('RGB')
    
    # ELA Preprocessing
    temp_path = '/tmp/temp_resaved.jpg' # Vercel allows writing to /tmp
    original.save(temp_path, 'JPEG', quality=90)
    resaved = Image.open(temp_path)
    
    ela_image = ImageChops.difference(original, resaved)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image.resize(image_size)

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    ela_img = prepare_image(file)
    
    # Predict
    img_array = np.array(ela_img) / 255.0
    img_array = img_array.reshape(-1, 128, 128, 3)
    
    prediction = model.predict(img_array)
    pred_val = float(prediction[0][0])
    
    status = "TAMPERED" if pred_val > 0.5 else "ORIGINAL"
    return jsonify({
        "status": status,
        "confidence": pred_val if status == "TAMPERED" else 1 - pred_val
    })
    