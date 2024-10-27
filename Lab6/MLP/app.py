from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Tạo ứng dụng Flask
app = Flask(__name__)

# Tải mô hình đã huấn luyện
model = load_model("mnist_mlp_model.h5")

def preprocess_image(image):
    # Chuyển đổi ảnh thành dạng grayscale và resize về 28x28
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image = np.array(image).astype("float32") / 255
    image = image.reshape(1, 28 * 28)
    return image

# Route trang chính
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Kiểm tra nếu file được tải lên
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            # Xử lý ảnh tải lên và dự đoán
            image = Image.open(file)
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            digit = np.argmax(prediction)
            return render_template("index.html", digit=digit)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
