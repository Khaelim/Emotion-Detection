import cv2
import numpy as np
import tensorflow as tf

def get_class_string_from_index(index):
   for class_string, class_index in valid_generator.class_indices.items():
      if class_index == index:
         return class_string

class_names=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

#mymodel = tf.keras.models.load_model('C:/Users/Khaelim/Python Projects/Emotion-Detection/Compiled/my_model.h5', compile=True)
mymodel = tf.keras.models.load_model('video_model_v2.h5', compile=True)
mymodel.compile(tf.keras.optimizers.Adam(), loss='mse')

# mymodel = tf.keras.models.load_model('C:/Users/Khaelim/Python Projects/Emotion-Detection/Compiled/saved_model.pb', compile=True)
# mymodel.compile(tf.keras.optimizers.Adam(), loss='mse')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('D:/Downloads/Ap_Creative_Stock_Header.jpg')

while (True):
    ret, frame = cap.read()
    if not ret:
        continue

    #cv2.imshow("yoop", frame)


    img = frame
    temp_img = np.zeros([48,48,3],dtype=np.uint8)  # MAYBE RENAME TO [48, 48, 1]
    tx = 0
    ty = 0

    faces = face_cascade.detectMultiScale(frame, 1.2, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = frame[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        temp_img = frame[y:y + h, x:x + w]
        tx = x
        ty = y

    temp_img1 = tf.image.resize(temp_img, [48, 48], preserve_aspect_ratio=True)
    resize_image = tf.reshape(temp_img1, [-1, 48, 48, 1])

    predictions = mymodel.predict(resize_image)
    cv2.putText(img, str(class_names[np.argmax(predictions[0])]), (tx - 10, ty - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))


    scale_percent = 75 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    #make prediction
    predictions = mymodel.predict(resize_image)

    print(str("Prediction Array: "))
    print(predictions[0])
    print(str("Prediction No: "))
    print(np.argmax(predictions[0]))

    pred = str(class_names[np.argmax(predictions[0])])

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow("yoop", frame)

    #print(mymodel.summary())


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()