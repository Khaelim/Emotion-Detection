#importing openCV lib
import cv2, tensorflow as tf
#simple face detection
haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
#defineing the size for a face
(width, height) = (48, 48)

#attempt to put the tf model in there
#tensorflowNet = cv2.dnn.Model('saved_model.pb')
mymodel = tf.keras.models.load_model('C:/Khaelim/ForProgramming/TFmodels/Facial_emote/')
print("model loaded?")
#mymodel.predict()
#Creating a video capture object
vid = cv2.VideoCapture(0)

#Loop frames to create video
while(True):
    #capture video frame
    ret, frame = vid.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        #Attempt to estimate emotion from model
        predictions = mymodel.predict_classes(face_resize)
        print(predictions)




    #display the frame that was captured
    cv2.imshow("Test", frame)






    #if q is pressed exit the loop
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

#release the video capture object
vid.release()
#destroy all windows
cv2.destroyAllWindows()
