# %load_ext tensorboard
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import PIL.Image
import datetime
from PIL import Image
import pathlib
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

dataset_dir = path = os.path.join('C:/Users/Khaelim/Python Projects/Datasets/AUDIO_DS_FINAL/', "saved_main_data")
main_dir = 'C:/Users/Khaelim/Python Projects/Datasets/AUDIO_AS_IMAGE/'
train_dir = 'C:/Users/Khaelim/Python Projects/Datasets/AUDIO_DS_FINAL/saved_train_data/'
test_dir = 'C:/Users/Khaelim/Python Projects/Datasets/AUDIO_DS_FINAL/saved_test_data/'
main_dir = pathlib.Path(main_dir)

main_dir = pathlib.Path(main_dir)
image_count = len(list(main_dir.glob('*/*.jpg')))

angry = list(main_dir.glob('anger/*'))
disgust = list(main_dir.glob('disgust/*'))
fear = list(main_dir.glob('fear/*'))
happy = list(main_dir.glob('happy/*'))
neutral = list(main_dir.glob('neutral/*'))
sad = list(main_dir.glob('sad/*'))
surprise = list(main_dir.glob('surprise/*'))

batch_size = 256
img_height = 128
img_width = 32
epochs = 128

main_ds = tf.keras.preprocessing.image_dataset_from_directory(
    main_dir,
    labels='inferred',
    color_mode='grayscale',
    class_names=['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
    seed=54321,
    image_size=(img_height, img_width),
    batch_size=batch_size)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  main_dir,
  labels='inferred',
  color_mode='grayscale',
  class_names=['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
  validation_split=0.2,
  subset="training",
  seed=54321,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  main_dir,
  labels='inferred',
  color_mode='grayscale',
  class_names=['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# tf.data.experimental.save(main_ds, dataset_dir)
# tf.data.experimental.save(train_ds, main_dir)
# tf.data.experimental.save(test_ds, main_dir)


# To cache the model for repeated training

train_ds = train_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Listing some data
# class_names = main_dir.class_names
# print(class_names)
# print(len(main_dir.class_names))


#number of clasifications
num_classes = 7
#defining the model
model = tf.keras.Sequential([
  #layers.Dense(7, input_shape=([48, 48])),
  layers.Conv2D(7, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 2, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 2, activation='relu'),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(64, activation='relu'),
  layers.Dense(num_classes)
])
#compileing the model
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

#Attempting to add tensorboard callbacks
log_dir = "C:/Khaelim/ForProgramming/TBlogs"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph='true', update_freq='epoch')


#training the model
model.fit(
  train_ds,
  validation_data=test_ds,
  batch_size=batch_size,
  steps_per_epoch=None,
  epochs=epochs,
  callbacks=[tensorboard_callback])

# Saving options for model

#tf.saved_model.save(model, "C:/Khaelim/ForProgramming/TFmodels/Facial_emote/")

# tf.keras.models.save_model(
#     model, "my_audio_model.pb", overwrite=True, include_optimizer=True, save_format='pb',
#     signatures=None, options=None
# )
model.save('my_audio_model_v5.h5')


print('Done')
