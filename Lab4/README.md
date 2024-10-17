1. Framwork sử dụng:
   - Pandas: Xử lý dữ liệu dưới dạng DataFrame.
   - NumPy: Xử lý tính toán ma trận và mảng.
   - Scikit-learn: Chia dữ liệu thành tập train/test.
2. Thuật toán sử dụng:
   2.1 Decision Tree Classifier (Cây quyết định):
      - Nguyên lý:
        + Cây quyết định phân chia dữ liệu thành các nhánh dựa trên đặc trưng nào giúp giảm thiểu độ bất thuần (impurity) cao nhất.
        + Mỗi nút trong cây đại diện cho một đặc trưng và một điều kiện tách, còn các lá của cây đại diện cho các dự đoán.
      - Tham số chính:
        + max_depth: Độ sâu tối đa của cây.
        + min_samples_split: Số mẫu tối thiểu để tách thêm một nhánh.
      - Ưu điểm:
        + Đơn giản và dễ hiểu.
        + Hiệu quả với các tập dữ liệu nhỏ.
      - Nhược điểm:
        + Dễ overfit khi cây quá sâu.
        + Nhạy cảm với dữ liệu nhiễu.
    2.2 Random Forest Classifier (Rừng ngẫu nhiên):
      - Nguyên lý:
        + Là một mô hình ensemble kết hợp nhiều cây quyết định. Mỗi cây được huấn luyện trên một mẫu bootstrap (ngẫu nhiên) của tập dữ liệu ban đầu.
        + Dự đoán cuối cùng là kết quả "bỏ phiếu đa số" (majority voting) từ các cây.
      - Tham số chính:
        + n_trees: Số cây trong rừng.
        + max_depth: Độ sâu tối đa của mỗi cây.
        + n_features: Số đặc trưng con được chọn ngẫu nhiên để xây dựng từng cây.
      - Ưu điểm:
        + Giảm thiểu overfitting nhờ việc kết hợp nhiều cây.
        + Ít nhạy cảm với nhiễu.
      - Nhược điểm:
        + Tốn thời gian và tài nguyên để huấn luyện.
3. Kết quả chạy thuật toán:

   
![image](https://github.com/user-attachments/assets/a9b39a38-a74b-4e0e-b5b7-fc763281ff88)

4. So sánh 2 kết quả Ta thấy được thuật toán Decision Tree Classifier cho ra tỉ lệ là 1.0 trong khi thuật toán Random Forest Classifier là 0.725
