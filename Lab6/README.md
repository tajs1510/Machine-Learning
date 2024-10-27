FORMULA: 
  1. Framework sử dụng: Đoạn mã trên sử dụng PyTorch, một thư viện mã nguồn mở phổ biến cho học sâu. PyTorch cung cấp các công cụ để dễ dàng thực hiện các tính toán tensor, xây dựng và huấn luyện các mô hình học sâu

  2. Thuật toán:
     - Mean Squared Error (MSE): Đây là hàm tính lỗi phổ biến trong các bài toán hồi quy. Nó tính bình phương của sai số giữa dự đoán và giá trị thực.
     - Binary Cross-Entropy Loss (BCE): Đây là hàm tính lỗi dùng cho bài toán phân loại nhị phân. Hàm này so sánh đầu ra dự đoán với nhãn thực của mẫu.
     - Cross Entropy Loss: Hàm lỗi này được sử dụng cho các bài toán phân loại đa lớp. Nó kết hợp softmax và negative log-likelihood để tính toán lỗi.
     - Các hàm kích hoạt:
       + Sigmoid: Hàm này nén đầu ra vào khoảng (0,1), phổ biến trong các lớp đầu ra nhị phân.
       + ReLU (Rectified Linear Unit): Một hàm kích hoạt thường dùng để giới hạn các giá trị âm về 0, giữ nguyên các giá trị dương.
       + Softmax: Hàm này chuyển đổi một tập hợp các giá trị thành xác suất, tổng các xác suất bằng 1. Đây là hàm phổ biến ở lớp đầu ra của các bài toán phân loại đa lớp.
       + Tanh: Một hàm kích hoạt khác nén đầu ra vào khoảng (-1, 1), thường được dùng trong các lớp ẩn của mạng nơron.
         
  3. Kết quả:
     
     ![image](https://github.com/user-attachments/assets/b40acda9-f8b3-4151-8cad-d425bc260c46)
