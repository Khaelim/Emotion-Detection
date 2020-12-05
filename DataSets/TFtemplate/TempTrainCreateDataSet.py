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

#train_ds = tf.data.experimental.load(data_dir, tf.TensorSpec(shape=())

# Temp fix

#dataset_dir = path = os.path.join('C:/Khaelim/ForProgramming/TFdatasets/', "saved_test_data")
main_dir = 'C:/Khaelim/ForProgramming/FERv1/FER13/full/'
train_dir = 'C:/Khaelim/ForProgramming/FERv1/FER13/train/'
test_dir = 'C:/Khaelim/ForProgramming/FERv1/FER13/test/'
main_dir = pathlib.Path(main_dir)
train_dir = pathlib.Path(train_dir)
test_dir = pathlib.Path(test_dir)
image_count = len(list(main_dir.glob('*/*.jpg')))

angry = list(main_dir.glob('angry/*'))
disgust = list(main_dir.glob('disgust/*'))
fear = list(main_dir.glob('fear/*'))
happy = list(main_dir.glob('happy/*'))
neutral = list(main_dir.glob('neutral/*'))
sad = list(main_dir.glob('sad/*'))
surprise = list(main_dir.glob('surprise/*'))

batch_size = 32
img_height = 48
img_width = 48


main_ds = tf.keras.preprocessing.image_dataset_from_directory(
  main_dir,
  labels='inferred',
  color_mode='grayscale',
  class_names=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir,
  labels='inferred',
  color_mode='grayscale',
  class_names=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_dir,
  labels='inferred',
  color_mode='grayscale',
  class_names=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

## end temp fix

train_ds = train_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


image_count = len(list(data_dir.glob('*/*.jpg')))


batch_size = 32
img_height = 48
img_width = 48

# class_names = train_ds.class_names
# print(class_names)
# print(len(train_ds.class_names))

#caching images to increase performance
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#number of clasifications
num_classes = 7
#defining the model
model = tf.keras.Sequential([
  #layers.Dense(7, input_shape=([48, 48])),
  layers.Conv2D(7, 2, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 2, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 2, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 2, activation='elu'),
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
log_dir = "C:\Khaelim\ForProgramming\TBlogs"# + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph='true', update_freq='epoch')


#training the model
model.fit(
  train_ds,
  validation_data=val_ds,
  batch_size=80,
  steps_per_epoch=None,
  epochs=50,
  callbacks=[tensorboard_callback])

#
#tf.saved_model.save(model, "C:/Khaelim/ForProgramming/TFmodels/Facial_emote/")

tf.keras.models.save_model(
    model, "test.pb", overwrite=True, include_optimizer=True, save_format='pb',
    signatures=None, options=None
)
model.save('my_model.pb')

