from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Tải tập dữ liệu Wine
wine = datasets.load_wine()
X = wine.data
y = wine.target

# Chia tập dữ liệu thành train và test với tỷ lệ 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Xây dựng mô hình KNN với k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = knn.predict(X_test)

# Tính toán và in ra độ chính xác, độ nhạy (recall), và độ chính xác (precision)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')  # Sử dụng 'macro' để tính recall trung bình trên các lớp
precision = precision_score(y_test, y_pred, average='macro')  # Tương tự cho precision

# In kết quả
print(f"Độ chính xác (Accuracy): {accuracy:.4f}")
print(f"Độ nhạy (Recall): {recall:.4f}")
print(f"Độ chính xác (Precision): {precision:.4f}")
