import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
# 載入 Fashion MNIST 數據集
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 數據預處理
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 創建卷積神經網絡（CNN）模型
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 編譯模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 定義早停回調函數以監控驗證集的損失
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# 訓練模型並使用早停回調函數(在訓練過程若一定週期沒有改善則停止訓練)
history = model.fit(x_train, y_train, batch_size=64, epochs=30, validation_split=0.2, callbacks=[early_stopping])

# 在測試集上評估模型
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"測試集上的準確率：{test_accuracy}")

# 輸出最後一個驗證集的損失
val_loss = history.history['val_loss'][-1]
print(f"最後一個驗證集的損失：{val_loss}")

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