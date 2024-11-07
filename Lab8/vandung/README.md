1. Framework:
   - scikit-learn (sklearn): Thư viện học máy được sử dụng rộng rãi, cung cấp các công cụ cho việc tiền xử lý dữ liệu, huấn luyện mô hình và đánh giá mô hình.
   - NumPy: Thư viện phổ biến để xử lý các phép toán số học, đặc biệt là với các mảng và ma trận. Nó được sử dụng để tính toán khoảng cách Euclidean và các phép toán khác.
   - Pandas: Thư viện mạnh mẽ để xử lý và phân tích dữ liệu, đặc biệt là dữ liệu dạng bảng (dataframe). Trong ứng dụng này, Pandas được sử dụng để hiển thị dữ liệu mẫu trên giao diện web.

2. Thuật toán:
    - K-Nearest Neighbors (KNN): Đây là thuật toán học máy phân loại dựa trên việc xác định các điểm dữ liệu gần nhất (neighbors) trong không gian đặc trưng và gán nhãn cho điểm kiểm tra (test point) dựa trên nhãn của các điểm gần nhất.
      + Hàm knn_predict trong mã sử dụng khoảng cách Euclidean để tính toán các điểm gần nhất từ tập huấn luyện và xác định nhãn của các điểm kiểm tra.
      + Giá trị k=5 trong hàm knn_predict chỉ ra rằng mô hình sẽ xem xét 5 điểm gần nhất để đưa ra dự đoán.
    - Khoảng cách Euclidean: Đây là một phương pháp phổ biến để đo lường sự tương đồng giữa hai điểm trong không gian đặc trưng. Hàm euclidean_distance trong mã tính toán khoảng cách giữa mỗi điểm kiểm tra và các điểm huấn luyện.
    - Ma trận nhầm lẫn (Confusion Matrix): Đây là một công cụ đánh giá mô hình phân loại. Ma trận này giúp đo lường số lượng dự đoán đúng (True Positive - TP, True Negative - TN) và sai (False Positive - FP, False Negative - FN).

3. Kết quả:

   ![image](https://github.com/user-attachments/assets/d4c44d80-d599-48b2-8701-6215c0e373fc)
