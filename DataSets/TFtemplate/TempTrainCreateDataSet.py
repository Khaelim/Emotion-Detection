#%load_ext tensorboard
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

print(tf.__version__)
data_dir = 'C:/Khaelim/ForProgramming/FERv1/FER13/train/'
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))

# angry = list(data_dir.glob('angry/*'))
# disgust = list(data_dir.glob('disgust/*'))
# fear = list(data_dir.glob('fear/*'))
# happy = list(data_dir.glob('happy/*'))
# neutral = list(data_dir.glob('neutral/*'))
# sad = list(data_dir.glob('sad/*'))
# surprise = list(data_dir.glob('surprise/*'))

batch_size = 32
img_height = 48
img_width = 48

# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="validation",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)
print(len(train_ds.class_names))

#caching images to increase performance
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#number of clasifications
num_classes = 7
#defining the model
model = tf.keras.Sequential([
  #layers.experimental.preprocessing.Rescaling(1./255),
  layers.Dense(6, input_shape=(48, 48, 3)),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='softmax'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='elu'),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
#compileing the model
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

#Attempting to add tensorboard
# log_dir = "C:\Khaelim\ForProgramming\TBlogs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


#training the model
model.fit(
  train_ds,
  validation_data=val_ds,
  batch_size=128,
  steps_per_epoch=None,
  epochs=35)

#
#tf.saved_model.save(model, "C:/Khaelim/ForProgramming/TFmodels/Facial_emote/")

tf.keras.models.save_model(
    model, "C:/Khaelim/ForProgramming/TFmodels/Facial_emote/test.h5", overwrite=True, include_optimizer=True, save_format='pb',
    signatures=None, options=None
)
model.save('my_model.h5')
model.predict(data_dir.glob('angry/S010_004_00000017.png'))
model.save('my_model.h5')
