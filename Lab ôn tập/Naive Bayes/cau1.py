from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Tải dữ liệu Iris
du_lieu_iris = load_iris()
X, y = du_lieu_iris.data, du_lieu_iris.target

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra với tỷ lệ 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình Naive Bayes
mo_hinh = GaussianNB()
mo_hinh.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_du_doan = mo_hinh.predict(X_test)

# Tính độ chính xác của mô hình
do_chinh_xac = accuracy_score(y_test, y_du_doan)
print(do_chinh_xac)
