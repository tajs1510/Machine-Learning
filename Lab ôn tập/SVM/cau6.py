from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Tải dữ liệu Digits
digits = datasets.load_digits()

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=42)

# Tạo mô hình SVM với kernel tuyến tính
svm_model = SVC(kernel='linear')

# Huấn luyện mô hình trên dữ liệu huấn luyện
svm_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = svm_model.predict(X_test)

# Tính toán độ chính xác
accuracy = accuracy_score(y_test, y_pred)

# In kết quả độ chính xác
print(f"Độ chính xác trên tập kiểm tra: {accuracy * 100:.2f}%")
