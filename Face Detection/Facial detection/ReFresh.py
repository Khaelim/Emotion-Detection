import cv2
import numpy as np
import tensorflow as tf

mymodel = tf.keras.models.load_model('C:/Khaelim/ForProgramming/TFmodels/my_model.h5', compile=True)
mymodel.compile(tf.keras.optimizers.Adam(), loss='mse')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('D:\Khaelim\Documents\ProgrammingProjects\Facial detection V2\Soul-Mates.AU.S01E01.WEB-DLx264-JIVE.mp4')

while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        continue

    img = frame
    temp_img = np.zeros([48,48,3],dtype=np.uint8)

    faces = face_cascade.detectMultiScale(frame, 1.2, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = frame[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        temp_img = frame[y:y + h, x:x + w]
    temp_img1 = tf.image.resize(temp_img, [48,48], preserve_aspect_ratio=True)

    print(temp_img1.get_shape())

    resize_image = tf.reshape(temp_img1, [-1, 48, 48, 3])

    cv2.imshow("yoop", img)
    print(mymodel.predict(resize_image))
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()