import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Đường dẫn tới thư mục chứa dữ liệu
train_dir = 'test'

# Chuẩn bị dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Sử dụng ResNet50 tiền huấn luyện
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Tinh chỉnh ResNet
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Thay Flatten bằng Global Average Pooling
x = Dense(128, activation='relu')(x)  # Lớp kết nối hoàn toàn
x = Dropout(0.5)(x)  # Thêm Dropout để giảm overfitting
predictions = Dense(10, activation='softmax')(x)  # Lớp đầu ra cho 10 lớp

# Tạo mô hình hoàn chỉnh
model = Model(inputs=base_model.input, outputs=predictions)

# Mở khóa toàn bộ các tầng trong ResNet để huấn luyện
for layer in base_model.layers:
    layer.trainable = True

# Biên dịch mô hình
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình 30 epochs
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator
)

# Lưu mô hình
model.save('resnet_cifar10_model_full_train.h5')

# Đánh giá mô hình
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")
