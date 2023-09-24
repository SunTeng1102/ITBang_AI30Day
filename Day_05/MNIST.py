# 引入必要的庫
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 載入MNIST數據集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 數據預處理
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 建立神經網絡模型
model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28, 1)))  # 將圖像展平成一維數組
model.add(layers.Dense(512, activation='relu'))  # 添加一個具有512個神經元的全連接層
model.add(layers.Dense(10, activation='softmax'))  # 添加一個具有10個神經元的全連接層（用於10個類別的分類）

# 編譯模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# 評估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)