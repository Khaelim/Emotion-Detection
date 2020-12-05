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


def create_model(name):
    model = tf.keras.models.Sequential[
        tf.keras.layers.Dense(6, input_shape=(48, 48, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='softmax'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='elu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(7)
        # tf.keras.layers.Flatten(input_shape=(28, 28)),
        # tf.keras.layers.Dense(512, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.Dense(10, activation='softmax')
    ]
    model = model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    # tf.saved_model.save(model, "C:/Khaelim/ForProgramming/TFmodels/Facial_emote/")

    tf.keras.models.save_model(
        model, (str(name) + '.h5)'), overwrite=True, include_optimizer=True,
        save_format='pb',
        signatures=None, options=None
    )
    model.save((str(name) + '.h5)'))

    return model
