KNN_BT1 :
 1. Framework : Pandas và NumPy cho các phép toán số.
 2. Thuật toán sử dụng:
  - k-NN (k-Nearest Neighbors)
    + Là một phương pháp phân loại không tham số, được sử dụng cho cả phân loại và hồi quy.
    + Phân loại một điểm dữ liệu dựa trên lớp chiếm ưu thế trong số k láng giềng gần nhất của nó trong không gian đặc trưng.

KNN_BT2:
 1. Framework : Pandas, NumPy
 2. Thuật toán sử dụng:
  - k-NN (k-Nearest Neighbors)
    + Là một phương pháp phân loại không tham số, được sử dụng cho cả phân loại và hồi quy.
    + Các điểm lân cận được xác định dựa trên khoảng cách giữa chúng, trong trường hợp này, sử dụng khoảng cách Euclidean (thực chất là tính toán căn bậc hai của tổng bình phương hiệu số giữa các điểm).

Centroid_Practice:
 1. Framework : Pandas, NumPy, Scikit-learn
 2. Thuật toán sử dụng:
  - Phân loại dựa trên khoảng cách Euclidean
    + Thuật toán phân loại này sử dụng khoảng cách giữa các điểm dữ liệu trong không gian n chiều để phân loại một điểm mới.
    + ![image](https://github.com/user-attachments/assets/8306e552-debc-49b4-93f4-28ae8be1ca78)
  - Phân nhóm (Group by):
    + Tính toán trung bình cho các đặc trưng trong tập huấn luyện theo lớp để xây dựng một mô hình cơ sở.
    + Sử dụng phương thức groupby của Pandas để nhóm dữ liệu theo lớp và tính toán trung bình cho các cột số.
