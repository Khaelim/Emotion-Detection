import cv2
import nightly as nightly
import numpy as np
import os
from PIL import Image
import pathlib
import tensorflow as tf
from tensorflow.python.keras.distribute.distribute_strategy_test import get_model

dataset_dir = path = os.path.join('C:/Khaelim/ForProgramming/TFdatasets/', "saved_test_data")

mymodel = tf.keras.models.load_model('C:/Khaelim/ForProgramming/TFmodels/my_model.h5', compile=True)
image = Image.open('C:/Users/Khaelim/Desktop/datasets/ck/CK+48/anger/S010_004_00000017.png')
mymodel.compile(tf.keras.optimizers.Adam(), loss='mse')
#loading a saved dataset
dataset_dir = path = os.path.join('C:/Khaelim/ForProgramming/TFdatasets/test', "saved_test_data")
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

acc = mymodel.evaluate(test_ds)
print('Restored model, accuracy: {:5.2f}%'.format(acc))
print(test_ds.class_names)

#print(mymodel.predict(test_ds).shape) #predict_on_batch
predictions = mymodel.predict(test_ds)
score = tf.nn.softmax(predictions[0])


mymodel.predict(image)
