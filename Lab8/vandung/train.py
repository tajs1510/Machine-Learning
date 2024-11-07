# train.py
import numpy as np
from sklearn.model_selection import train_test_split

# Tạo dữ liệu giả định cho KNN
np.random.seed(42)
data_size = 1000
X_class0 = np.random.multivariate_normal([2, 2], [[1.5, 0.75], [0.75, 1.5]], data_size // 2)
X_class1 = np.random.multivariate_normal([4, 4], [[1.5, 0.75], [0.75, 1.5]], data_size // 2)
X = np.vstack((X_class0, X_class1))
y = np.hstack((np.zeros(data_size // 2), np.ones(data_size // 2)))

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra với test_size = 0.3 và random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
