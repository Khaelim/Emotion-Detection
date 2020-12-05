import os

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

print(tf.version.VERSION)

ds = tfds.load('video_emotion', split='train', shuffle_files=True)
assert isinstance(ds, tf.data.Dataset)
print(ds)

ds, info = tfds.load('video_emotion', split='train', with_info=True, data_dir= 'C:/Khaelim/ForProgramming/FERv1/FER13/train/',)

tfds.as_dataframe(ds.take(4), info)

builder = tfds.builder('video_emotion')
info = builder.info

# slice the dataset

train_ds, test_ds = tfds.load('video_emotion', split=[
    tfds.core.ReadInstruction('train'),
    tfds.core.ReadInstruction('test'),
])

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.l()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 48 * 48) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 48) / 255.0

# print(info)
# print(info.features["label"].num_classes)
# print(info.features["label"].names)
# print(info.features["label"].int2str(7))  # Human readable version (8 -> 'cat')
# print(info.features["label"].str2int('7'))
# print(info.features.shape)
# print(info.features.dtype)
# print(info.features['image'].shape)
# print(info.features['image'].dtype)
# print(info.splits)
# print(list(info.splits.keys()))

# checkpoint
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# load the model
model = tf.keras.models.load_model('saved_model/my_model')
model.summary()

model.fit(train_images,
          train_labels,
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Pass callback to training