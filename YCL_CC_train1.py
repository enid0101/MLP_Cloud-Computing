import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical


# 替代舊的'__file__'用法，若在notebook中運行
root_path = os.getcwd()

# 設定超參數
learning_rate = 0.01
epochs = 5000
batch_size = 32

def main():
    # TensorBoard日誌設定
    #log_dir = os.path.join(root_path, "logs/YOURMODEL")
    #tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

    # 創建神經網絡模型
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(25,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(5, activation='softmax')
    ])

    # 編譯模型
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 載入訓練數據
    train_dataset = np.load('D:/E/dataset/train.npz')
    train_data = train_dataset['data']
    train_labels = to_categorical(train_dataset['label'])

    # 訓練模型
    model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs)

    # 儲存模型
    model.save(os.path.join(root_path, 'YOURMODEL.h5'))

    # 驗證模型
    valid_dataset = np.load('D:/E/dataset/validation.npz')
    valid_data = valid_dataset['data']
    valid_labels = to_categorical(valid_dataset['label'])

    # 預測並計算準確率
    predicted_labels = model.predict(valid_data)
    predicted_labels = np.argmax(predicted_labels, axis=1)
    true_labels = np.argmax(valid_labels, axis=1)

    print(predicted_labels)
    print(true_labels)
    accuracy = np.mean(true_labels == predicted_labels)
    print(f'Accuracy: {accuracy}')

if __name__ == "__main__":
    main()
