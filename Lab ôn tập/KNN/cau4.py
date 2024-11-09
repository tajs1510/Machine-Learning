import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Tải tập dữ liệu Wine
wine = datasets.load_wine()
X = wine.data
y = wine.target

# Chia tập dữ liệu thành train và test với tỷ lệ 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Các giá trị k khác nhau
k_values = [1, 3, 7, 9]
accuracy_scores = []

# Thử nghiệm với các giá trị k
for k in k_values:
    # Xây dựng mô hình KNN với k hiện tại
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Dự đoán và tính toán độ chính xác
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Vẽ đồ thị
plt.plot(k_values, accuracy_scores, marker='o')
plt.title('Mối quan hệ giữa k và độ chính xác của mô hình KNN')
plt.xlabel('Giá trị k')
plt.ylabel('Độ chính xác')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# In kết quả độ chính xác tương ứng với các giá trị k
for k, accuracy in zip(k_values, accuracy_scores):
    print(f"Độ chính xác với k={k}: {accuracy:.4f}")
