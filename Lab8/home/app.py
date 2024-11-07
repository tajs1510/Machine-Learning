from flask import Flask, render_template
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Các giá trị từ mô hình KNN
accuracy = 0.93  # Ví dụ
recall = 0.92    # Ví dụ
precision = 0.94 # Ví dụ

@app.route("/")
def index():
    # Tạo biểu đồ trực quan hóa các chỉ số hiệu năng
    metrics = ['Accuracy', 'Recall', 'Precision']
    values = [accuracy, recall, precision]

    # Biểu đồ cột
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values, color=['blue', 'green', 'red'])
    plt.ylim(0, 1)
    plt.title("Metrics of KNN Model")
    plt.xlabel("Metrics")
    plt.ylabel("Values")

    # Lưu biểu đồ vào buffer và mã hóa để hiển thị trong HTML
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return render_template("index.html", plot_url=plot_url, accuracy=accuracy, recall=recall, precision=precision)

if __name__ == "__main__":
    app.run(debug=True)

