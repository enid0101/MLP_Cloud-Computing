from tensorflow import keras
import gymnasium as gym
import numpy as np
import os
from keras.utils import to_categorical



root_path = os.path.abspath(os.path.dirname(__file__))


def main():
  total_reward = 0
  #Loading model
  model = keras.models.load_model('D:/E/TEST/YOURMODEL.h5')
  env = gym.make('highway-fast-v0', render_mode='rgb_array')

  # for _ in range(1):
  #   (obs,info) = env.reset()
  #   done = False
  #   truncated = False
  #   while not (done or truncated):
  #     env.render()
  #     obs = obs.reshape(1,25)
  #     action = model.predict(obs)
  #     action = np.argmax(action, axis=1)
  #     obs, reward, done, truncated, info = env.step(int(action))
  #     total_reward += reward
  #     val_dataset_path = os.path.join(root_path, 'dataset', 'test.npz')
  val_dataset = np.load("D:/E/TEST/test.npz")
  val_label = val_dataset['label']
  val_data = val_dataset['data']


  #processing label with one-hot-encoding
  val_label = to_categorical(val_label, num_classes=5)
  predictions = model.predict(val_data)
  predicted_classes = np.argmax(predictions, axis=1)
  # Calculating accuracy
  true_labels = np.argmax(val_label, axis=1)
  print(predicted_classes)
  print(true_labels)
  accuracy = np.mean(true_labels == predicted_classes)
  print(f'Accuracy: {accuracy}')
      
  return int(total_reward)



if __name__ == "__main__":
  rewards = []
  for round in range(0,10):
    reward = main()
    rewards.append(reward)   
  print(rewards)