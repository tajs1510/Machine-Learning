import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Tải dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Chuẩn hóa dữ liệu ảnh
x_train, x_test = x_train / 255.0, x_test / 255.0

# Chuyển đổi nhãn thành dạng one-hot
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Xây dựng mô hình MLP với hai tầng ẩn
model = Sequential([
    Flatten(input_shape=(28, 28)),       # Chuyển ảnh từ 2D (28x28) thành 1D
    Dense(128, activation='relu'),       # Tầng ẩn thứ nhất với 128 nơ-ron
    Dense(64, activation='relu'),        # Tầng ẩn thứ hai với 64 nơ-ron
    Dense(10, activation='softmax')      # Tầng đầu ra với 10 nơ-ron (số lớp là 10 vì có 10 chữ số từ 0 đến 9)
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Đánh giá độ chính xác trên tập kiểm tra
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("Độ chính xác trên tập kiểm tra:", test_accuracy)
