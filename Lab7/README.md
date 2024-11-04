1. Framework sử dụng: 
  - Pytorch: Thư viện học sâu mạnh mẽ giúp xây dựng và huấn luyện các mô hình học máy, cung cấp các lớp và hàm để dễ dàng phát triển mạng nơ-ron.
  - Numpy: Thư viện cơ bản hỗ trợ tính toán ma trận và mảng, hữu ích cho các phép toán số học và thống kê trong quá trình xử lý dữ liệu.
  - Matplotlib: Thư viện dùng để trực quan hóa kết quả, bao gồm biểu đồ mất mát và độ chính xác của mô hình qua các epoch.
  - Torchvision: Thư viện cung cấp các bộ dữ liệu chuẩn và các công cụ biến đổi dữ liệu hình ảnh, hỗ trợ tải và xử lý các tập dữ liệu hình ảnh như CIFAR-10.

2. Thuật toán:
   MLP (Multi-Layer Perceptron) : Xây dựng một mô hình MLP để phân loại hình ảnh trong tập dữ liệu CIFAR-10, nhận diện chính xác từng loại hình ảnh, có các bước như sau:
    - Tiền xử lý dữ liệu: Tải và chuẩn hóa dữ liệu về khoảng từ 0 đến 1 để phù hợp với mô hình.
    - Xây dựng mô hình MLP: Sử dụng các lớp Linear, hàm kích hoạt ReLU, và lớp đầu ra với 10 nút cho các lớp phân loại.
    - Huấn luyện mô hình: Sử dụng phương pháp tối ưu SGD và hàm mất mát CrossEntropy.
    - Đánh giá mô hình: Kiểm tra độ chính xác và mất mát trên tập dữ liệu kiểm tra sau mỗi epoch.
3. Kết quả:
   ![image](https://github.com/user-attachments/assets/41176f7f-bf59-43b2-8ada-128fe800d7c5)

   -![image](https://github.com/user-attachments/assets/b1e17e50-4529-4efb-8830-a46447d9aa60)

