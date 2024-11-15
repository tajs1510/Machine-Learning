from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64

from PIL import Image
import io

app = Flask(__name__)

# Load the pre-trained CNN model
model = load_model('resnet_model.h5')

# List of CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Prepare image for prediction
def prepare_image(img):
    img_resized = img.resize((32, 32))  # Resize for model input
    img_array = np.array(img_resized) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Open image for display and processing
    original_img = Image.open(file.stream)
    
    # Process image for model prediction
    img_for_model = prepare_image(original_img)
    
    # Predict with the model
    prediction = model.predict(img_for_model)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_name = class_names[predicted_class]
    confidence = float(prediction[0][predicted_class])

    # Convert the original high-quality image to base64 for display
    img_byte_arr = io.BytesIO()
    original_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    img_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')

    return render_template('index.html', image_data=img_base64, class_name=class_name, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
