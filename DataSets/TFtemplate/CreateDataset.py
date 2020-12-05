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
train_dir = 'C:/Khaelim/ForProgramming/FERv1/FER13/train/'
val_dir = 'C:/Khaelim/ForProgramming/FERv1/FER13/test/'
train_dir = pathlib.Path(train_dir)
val_dir = pathlib.Path(val_dir)

image_count = len(list(train_dir.glob('*/*.jpg')))

angry = list(train_dir.glob('angry/*'))
angry = list(val_dir.glob('angry/*'))
disgust = list(train_dir.glob('disgust/*'))
disgust = list(val_dir.glob('disgust/*'))
fear = list(train_dir.glob('fear/*'))
fear = list(val_dir.glob('fear/*'))
happy = list(train_dir.glob('happy/*'))
happy = list(val_dir.glob('happy/*'))
neutral = list(train_dir.glob('neutral/*'))
neutral = list(val_dir.glob('neutral/*'))
sad = list(train_dir.glob('sad/*'))
sad = list(val_dir.glob('sad/*'))
surprise = list(train_dir.glob('surprise/*'))
surprise = list(val_dir.glob('surprise/*'))

batch_size = 32
img_height = 48
img_width = 48


test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  val_dir,
  labels='inferred',
  color_mode='grayscale',
  subset="validation",
  class_names=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
  seed=123,
  image_size=(img_height, img_width),
  validation_split=0.2,
  batch_size=batch_size)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  val_dir,
  labels='inferred',
  color_mode='grayscale',
  subset="training",
  class_names=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
  seed=123,
  image_size=(img_height, img_width),
  validation_split=0.2,
  batch_size=batch_size)

tf.data.experimental.save(train_ds, dataset_dir)
tf.data.experimental.save(test_ds, dataset_dir)



#To cache the model for repeated training
#test_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)


print('Done')
