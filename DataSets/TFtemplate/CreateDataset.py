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

dataset_dir = path = os.path.join('C:/Khaelim/ForProgramming/TFdatasets/', "saved_test_data")
main_dir = 'C:/Khaelim/ForProgramming/FERv1/FER13/full/'
train_dir = 'C:/Khaelim/ForProgramming/FERv1/FER13/train/'
test_dir = 'C:/Khaelim/ForProgramming/FERv1/FER13/test/'
main_dir = pathlib.Path(main_dir)

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
tf.data.experimental.save(train_ds, train_dir)
tf.data.experimental.save(test_ds, test_dir)



#To cache the model for repeated training
test_ds = train_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


print('Done')
