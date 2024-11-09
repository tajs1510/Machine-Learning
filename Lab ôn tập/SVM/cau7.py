import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Tải dữ liệu Digits
digits = datasets.load_digits()

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=42)

# Hàm huấn luyện và đánh giá mô hình
def train_and_evaluate(kernel_type):
    model = SVC(kernel=kernel_type)
    
    # Đo thời gian huấn luyện
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)
    
    # Tính độ chính xác
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, training_time

# Thử nghiệm với các kernel khác nhau
kernels = ['linear', 'rbf', 'poly']
results = {}

for kernel in kernels:
    accuracy, training_time = train_and_evaluate(kernel)
    results[kernel] = {'accuracy': accuracy, 'training_time': training_time}

# In kết quả
for kernel, result in results.items():
    print(f"Kernel: {kernel}")
    print(f"  Độ chính xác: {result['accuracy'] * 100:.2f}%")
    print(f"  Thời gian huấn luyện: {result['training_time']:.4f} giây")
    print("-" * 40)
