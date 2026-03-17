from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS 
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import tensorflow as tf
import io, base64, os

# 1. SETUP ABSOLUTE PATHS (The Fix)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC_DIR = os.path.join(BASE_DIR, 'public')
MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')

app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path='')
CORS(app) 

# 2. LOAD MODEL
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"✅ SUCCESS: Model loaded from {MODEL_PATH}")
    else:
        print(f"❌ ERROR: model.h5 not found at {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Model Load Error: {e}")

# --- FORENSIC LOGIC ---
def convert_to_ela(image, quality=90):
    original = image.convert('RGB')
    buffer = io.BytesIO()
    original.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    temporary = Image.open(buffer)
    diff = ImageChops.difference(original, temporary)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale = 255.0 / max_diff
    return ImageEnhance.Brightness(diff).enhance(scale)

# --- ROUTES ---
@app.route('/')
def home():
    # If this still 404s, it means the filename isn't index.html
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    img = Image.open(file.stream)
    
    # AI Processing
    ela_img = convert_to_ela(img)
    processed = ela_img.resize((128, 128))
    img_array = np.array(processed) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if model:
        pred = float(model.predict(img_array)[0][0])
        is_t = bool(pred > 0.5)
        conf = pred if is_t else (1.0 - pred)
        
        buffered = io.BytesIO()
        ela_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'status': 'Tampered' if is_t else 'Authentic',
            'confidence': round(conf * 100, 2),
            'ela_image_base64': f"data:image/jpeg;base64,{img_str}"
        })
    return jsonify({'error': 'Model not loaded'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)