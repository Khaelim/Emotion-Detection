#%load_ext tensorboard
import tempfile

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

"""Load test, Currently performed below"""
# train_ds=tf.data.experimental.load(train_dir,
#     tf.TensorSpec(shape=([48,48]), dtype=tf.float32))
#
# test_ds=tf.data.experimental.load(test_dir,
#     tf.TensorSpec(shape=([48,48]), dtype=tf.float32))

dataset_dir = path = os.path.join('C:/Users/Khaelim/Python Projects/Datasets/IMAGE_FINAL/', "saved_test_data")
# main_dir = 'C:/Users/Khaelim/Python Projects/Datasets/IMAGE_FINAL/'
# main_dir = pathlib.Path(main_dir)

# train_dir = 'C:/Khaelim/ForProgramming/FERv1/FER13/train/'
# test_dir = 'C:/Khaelim/ForProgramming/FERv1/FER13/test/'
# test_dir = 'D:\Khaelim\Documents\ProgrammingProjects\Emotion-Detection\DataSets\Tests'

# train_dir = pathlib.Path(train_dir)
# test_dir = pathlib.Path(test_dir)
# image_count = len(list(main_dir.glob('*/*.jpg')))

# angry = list(main_dir.glob('anger/*'))
# disgust = list(main_dir.glob('disgust/*'))
# fear = list(main_dir.glob('fear/*'))
# happy = list(main_dir.glob('happy/*'))
# neutral = list(main_dir.glob('neutral/*'))
# sad = list(main_dir.glob('sad/*'))
# surprise = list(main_dir.glob('surprise/*'))

"""Creating dataset in done separately"""
# main_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   main_dir,
#   labels='inferred',
#   color_mode='grayscale',
#   class_names=['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   train_dir,
#   labels='inferred',
#   color_mode='grayscale',
#   class_names=['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)
#
#
# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   test_dir,
#   labels='inferred',
#   color_mode='grayscale',
#   class_names=['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

# class_names = train_ds.class_names
# print(class_names)
# print(len(train_ds.class_names))

#caching images to increase performance
# AUTOTUNE = tf.data.experimental.AUTOTUNE

# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# train_ds = train_dir.cache().prefetch(buffer_size=AUTOTUNE)

"""Loading the dataset"""
main_dataset = tf.data.experimental.load(path)
"""Caching the dataset"""
AUTOTUNE = tf.data.experimental.AUTOTUNE

main_dataset = main_dataset.cache().prefetch(buffer_size=AUTOTUNE)


#number of clasifications
num_classes = 7
#defining the model
model = tf.keras.Sequential([
  #layers.experimental.preprocessing.Rescaling(1./255),
  layers.Dense(6, input_shape=(48, 48, 1)),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
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

log_dir = "C:/Khaelim/ForProgramming/TBlogs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Create some callbacks
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True),
             tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=15, verbose=1,
                                                  min_lr=0.000001),
             tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]


#training the model in loop w/ checkpoint

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='C:/Khaelim/ForProgramming/checkpooints',
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)


batch_size = 32
img_height = 48
img_width = 48
epochs = 100

model.fit(
  main_dataset,
  #validation_data=test_ds,
  batch_size=batch_size,
  steps_per_epoch=None,
  epochs=epochs,
  callbacks=callbacks)

#
tf.saved_model.save(model, "./test1.h5")

tf.keras.models.save_model(
    model, "test.h5", overwrite=True, include_optimizer=True, save_format='pb',
    signatures=None, options=None
)

model.save('my_model_v2.h5')
