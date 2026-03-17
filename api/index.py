from flask import Flask, request, jsonify
from flask_cors import CORS
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io
import base64

app = Flask(__name__)
CORS(app)

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def prepare_image(image):
    # Same ELA logic as before...
    temp_io = io.BytesIO()
    image.save(temp_io, 'JPEG', quality=90)
    temp_io.seek(0)
    temp_img = Image.open(temp_io)
    ela_img = ImageChops.difference(image, temp_img)
    # ... (Keep your existing ELA enhancement logic here)
    return ela_img.resize((128, 128))

@app.route('/api/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    img = Image.open(file).convert('RGB')
    ela_img = prepare_image(img)
    
    # Prepare input for TFLite
    input_data = np.array(ela_img, dtype=np.float32).flatten() / 255.0
    input_data = input_data.reshape(input_details[0]['shape'])
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    status = "Fake" if prediction > 0.5 else "Real"
    return jsonify({
        "status": status,
        "confidence": f"{prediction*100:.2f}%" if status == "Fake" else f"{(1-prediction)*100:.2f}%"
    })

if __name__ == '__main__':
    app.run()