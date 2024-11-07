# app.py
from flask import Flask, render_template
import numpy as np
import pandas as pd
from train import X_train, X_test, y_train, y_test

app = Flask(__name__)

# Hàm tính khoảng cách Euclidean
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Hàm dự đoán KNN
def knn_predict(X_train, y_train, X_test, k=5):
    y_pred = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, x) for x in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
        y_pred.append(most_common)
    return np.array(y_pred)

# Hàm ma trận nhầm lẫn (confusion matrix)
def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])

# Hàm đánh giá mô hình
def evaluate_model(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) 
    specificity = TN / (TN + FP) 
    precision = TP / (TP + FP) 
    f1 = 2 * precision * recall / (precision + recall) 
    return {
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
    }

@app.route('/')
def index():
    # Hiển thị mẫu dữ liệu
    sample_data = pd.DataFrame(X_train[:5], columns=["Feature 1", "Feature 2"]).to_html(index=False)
    sample_labels = y_train[:5]

    # Dự đoán trên tập kiểm tra với k = 5
    y_pred_knn = knn_predict(X_train, y_train, X_test, k=5)
    evaluation_results = evaluate_model(y_test, y_pred_knn)

    return render_template('index.html', sample_data=sample_data, sample_labels=sample_labels, evaluation_results=evaluation_results)

if __name__ == '__main__':
    app.run(debug=True)
