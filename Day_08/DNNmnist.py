import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 載入MNIST數據集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# 數據預處理
train_images = train_images.reshape((60000, 28 * 28))  # 攤平成一維
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 載入 MNIST 數據集
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 數據預處理
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# 將標籤進行獨熱編碼
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# 建立深度神經網路模型
model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28, 1)))  # 將圖像展平成一維數組
model.add(layers.Dense(128, activation='relu'))  # 添加第一個全連接層，128個神經元，ReLU激活函數
model.add(layers.Dense(64, activation='relu'))   # 添加第二個全連接層，64個神經元，ReLU激活函數
model.add(layers.Dense(10, activation='softmax'))  # 輸出層，10個神經元，softmax激活函數，用於10個類別的分類

# 編譯模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型並記錄損失
loss_history = []  # 初始化一個空列表來保存訓練過程中的損失值


# 訓練模型，並記錄每個 epoch 的損失值
history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))


# 評估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('測試準確率:', test_acc)
# 建立 LOSS 圖表
#train loss
plt.plot(history.history['loss'])
#test loss
plt.plot(history.history['val_loss'])
#標題
plt.title('Model loss')
#y軸標籤
plt.ylabel('Loss')
#x軸標籤
plt.xlabel('Epoch')
#顯示折線的名稱
plt.legend(['Train', 'Test'], loc='upper left')
#顯示折線圖
plt.show()

# 評估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
