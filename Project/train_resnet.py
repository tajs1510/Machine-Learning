import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Hàm xây dựng một ResNet block
def resnet_block(inputs, filters, strides=1):
    # Nhánh chính
    x = Conv2D(filters, (3, 3), strides=strides, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), strides=1, padding="same")(x)
    x = BatchNormalization()(x)

    # Nhánh shortcut
    if strides != 1 or inputs.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=strides, padding="same")(inputs)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = inputs

    # Kết hợp nhánh chính và shortcut
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# Xây dựng ResNet
def build_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Tầng đầu vào
    x = Conv2D(64, (3, 3), strides=1, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Các ResNet blocks
    x = resnet_block(x, 64, strides=1)
    x = resnet_block(x, 64, strides=1)

    x = resnet_block(x, 128, strides=2)
    x = resnet_block(x, 128, strides=1)

    x = resnet_block(x, 256, strides=2)
    x = resnet_block(x, 256, strides=1)

    # Tầng đầu ra
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Tạo mô hình
    model = Model(inputs, outputs)
    return model

# Đường dẫn tới thư mục chứa dữ liệu
train_dir = 'test'

# Chuẩn bị dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
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

# Xây dựng mô hình ResNet
input_shape = (32, 32, 3)
num_classes = 10
model = build_resnet(input_shape, num_classes)

# Biên dịch mô hình
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator
)

# Lưu mô hình
model.save('resnet_model.h5')

# Đánh giá mô hình
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")
