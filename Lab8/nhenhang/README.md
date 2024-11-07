1. Framemwork sử dụng: không có
2. Thuật toán sử dụng: Ma trận nhầm lẫn (Confusion Matrix): Các giá trị TP, TN, FP, và FN đại diện cho số lượng True Positive, True Negative, False Positive, và False Negative của mô hình. Dựa trên các giá trị này, các chỉ số hiệu suất được tính toán như sau:
 - Accuracy (Độ chính xác): (TP + TN) / (TP + TN + FP + FN) - tỷ lệ dự đoán đúng trên tổng số dự đoán.
 - Recall (Độ nhạy): TP / (TP + FN) - tỷ lệ các mẫu dương tính được dự đoán đúng.
 - Specificity (Độ đặc hiệu): TN / (TN + FP) - tỷ lệ các mẫu âm tính được dự đoán đúng.
 - Precision (Độ chính xác): TP / (TP + FP) - tỷ lệ các mẫu dương tính trong các mẫu được dự đoán là dương tính.
 - F1 Score: 2 * (Precision * Recall) / (Precision + Recall) - trung bình điều hòa của Precision và Recall, dùng để đánh giá mô hình khi có sự mất cân bằng lớp.
3. Kết quả:
   
  ![image](https://github.com/user-attachments/assets/8fd508f0-647e-4230-8bff-a17737f920b5)

