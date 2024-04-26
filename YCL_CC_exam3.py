from tensorflow import keras
import gymnasium as gym
import numpy as np
import os
from keras.utils import to_categorical

root_path = os.path.abspath(os.path.dirname(__file__))

def main():
    model = keras.models.load_model('D:/E/TEST/YOURMODEL.h5')
    val_dataset = np.load("D:/E/TEST/test.npz")
    val_data = val_dataset['data']
    val_label = to_categorical(val_dataset['label'], num_classes=5)
    
    train_dataset = np.load("D:/E/TEST/train.npz")
    train_data = train_dataset['data']
    train_labels = to_categorical(train_dataset['label'], num_classes=5)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(
        train_data,
        train_labels,
        batch_size=32,
        epochs=10,
        validation_data=(val_data, val_label)
    )
    
    model.save('D:/E/TEST/MODEL_UPDATED.h5')
    
    predictions = model.predict(val_data)
    predicted_classes = np.argmax(predictions, axis=1)
    true_labels = np.argmax(val_label, axis=1)
    accuracy = np.mean(true_labels == predicted_classes)
    print(f'Accuracy: {accuracy}')

if __name__ == "__main__":
    main()
