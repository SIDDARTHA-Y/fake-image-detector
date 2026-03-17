import tensorflow as tf

# 1. Load your existing model
print("Loading H5 model...")
model = tf.keras.models.load_model('model.h5')

# 2. Setup the converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Convert the model
print("Converting... this may take a minute.")
tflite_model = converter.convert()

# 4. Save the new version
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Success! Your 'model.tflite' file is now in your folder.")