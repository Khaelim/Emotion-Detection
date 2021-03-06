import tkinter as tk
import cv2
import numpy as np
import tensorflow as tf

def open_video():
    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    mymodel = tf.keras.models.load_model('D:\Documents\Projects-DESKTOP-QOPQM8O\Python\Emotion-Detection\Menu\my_model.h5', compile=True)
    mymodel.compile(tf.keras.optimizers.Adam(), loss='mse')

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

    while (True):
        ret, frame = cap.read()
        if not ret:
            continue

        # cv2.imshow("yoop", frame)

        img = frame
        temp_img = np.zeros([48, 48, 3], dtype=np.uint8)
        x = 0
        y = 0

        faces = face_cascade.detectMultiScale(frame, 1.2, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = frame[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            temp_img = frame[y:y + h, x:x + w]
            #tx = x
            #ty = y

            temp_img = tf.image.resize(temp_img, [48, 48], preserve_aspect_ratio=True)
            temp_img = tf.reshape(temp_img, [-1, 48, 48, 3])

            predictions = mymodel.predict(temp_img)
            cv2.putText(img, str(class_names[np.argmax(predictions[0])]), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 255, 0))

        scale_percent = 75  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # make prediction
        # predictions = mymodel.predict(temp_img)
        #
        # print(str("Prediction Array: "))
        # print(predictions[0])
        # print(str("Prediction No: "))
        # print(np.argmax(predictions[0]))

        #pred = str(class_names[np.argmax(predictions[0])])

        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow("yoop", resized)

        # print(mymodel.summary())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def open_audio():
    print("ToDo")


def open_audio_video():
    print("ToDo")


root = tk.Tk()
frame = tk.Frame(root)
frame.pack()

video = tk.Button(frame,
                   text="Video",
                   command=open_video)
video.pack(side=tk.LEFT)

audio = tk.Button(frame,
                    text="Audio",
                    command=open_audio)
audio.pack(side=tk.LEFT)

audioVideo = tk.Button(frame,
                    text="Audio + Video",
                    command=open_audio_video)
audioVideo.pack(side=tk.LEFT)

quit = tk.Button(frame,
                   text="Quit",
                   fg="red",
                   command=quit)
quit.pack(side=tk.LEFT)

root.mainloop()