#%load_ext tensorboard
import tensorflow as tf

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

(x_shape, y_shape) = (48, 48)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(x_shape, y_shape)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=x_shape,
          y=y_shape,
          epochs=5,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback])