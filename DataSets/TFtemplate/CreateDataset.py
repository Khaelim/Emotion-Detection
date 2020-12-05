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
data_dir = 'C:/Khaelim/ForProgramming/FERv1/FER13/test/'
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))

angry = list(data_dir.glob('angry/*'))
disgust = list(data_dir.glob('disgust/*'))
fear = list(data_dir.glob('fear/*'))
happy = list(data_dir.glob('happy/*'))
neutral = list(data_dir.glob('neutral/*'))
sad = list(data_dir.glob('sad/*'))
surprise = list(data_dir.glob('surprise/*'))

batch_size = 32
img_height = 48
img_width = 48


test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  labels='inferred',
  color_mode='grayscale',
  class_names=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

tf.data.experimental.save(test_ds, dataset_dir)

print('Done')
