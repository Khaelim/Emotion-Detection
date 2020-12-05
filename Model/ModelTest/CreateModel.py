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
train_dir = 'C:\Khaelim\ForProgramming\FERv1\FER13\\train\\11033984163767626638\saved_test_data'
test_dir = 'C:\Khaelim\ForProgramming\FERv1\FER13\\test\\11033984163767626638\saved_test_data'
train_dir = pathlib.Path(train_dir)
test_dir = pathlib.Path(test_dir)

#image_count = len(list(train_dir.glob('*/*.jpg')))

batch_size = 32
img_height = 48
img_width = 48

train_ds=tf.data.experimental.load(train_dir,
    tf.TensorSpec(shape=([48,48]), dtype=tf.float32))

test_ds=tf.data.experimental.load(test_dir,
    tf.TensorSpec(shape=([48,48]), dtype=tf.float32))

class_names = train_ds.class_names
print(class_names)
print(len(train_ds.class_names))

#caching images to increase performance
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = test_dir.cache().prefetch(buffer_size=AUTOTUNE)

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
log_dir = "C:\Khaelim\ForProgramming\TBlogs" #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


#training the model in loop w/ checkpoint

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='D:\Khaelim\Documents\ProgrammingProjects\Emotion-Detection\checkpooints',
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)


model.fit(
  train_ds,
  validation_data=test_ds,
  batch_size=128,
  steps_per_epoch=None,
  epochs=35,
  callbacks=[tensorboard_callback])

#
#tf.saved_model.save(model, "C:/Khaelim/ForProgramming/TFmodels/Facial_emote/")

tf.keras.models.save_model(
    model, "C:/Khaelim/ForProgramming/TFmodels/Facial_emote/test.h5", overwrite=True, include_optimizer=True, save_format='pb',
    signatures=None, options=None
)

model.save('my_model.h5')
