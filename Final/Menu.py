import tkinter as tk
import cv2
import librosa
import numpy as np
import pyaudio
import tensorflow as tf
import keyboard

# Class labels
from matplotlib import pyplot as plt

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Video set-up
video_model = tf.keras.models.load_model('image_model_v5.h5', compile=True)
video_model.compile(tf.keras.optimizers.Adam(), loss='mse')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Audio set-up
audio_model = tf.keras.models.load_model('my_audio_model_v5.h5', compile=True)
audio_model.compile(tf.keras.optimizers.Adam(), loss='mse')
sample_len = 16000
frame_len = int(16000 * 1)  # 1sec


def open_video():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # cv2.imshow("yoop", frame)

        img = frame
        temp_img = np.zeros([48, 48, 3], dtype=np.uint8)
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

        # make prediction
        video_predictions = video_model.predict(resize_image)
        cv2.putText(img, str(class_names[np.argmax(video_predictions[0])]), (tx - 10, ty - 10), cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0))

        scale_percent = 75  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        print(str("Prediction Array: "))
        print(video_predictions[0])
        print(str("Prediction No: "))
        print(np.argmax(video_predictions[0]))

        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow("yoop", resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def open_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_len,
                    input=True,
                    frames_per_buffer=frame_len)
    plt.ion()

    while True:
        # data read
        data = stream.read(frame_len, exception_on_overflow=False)
        # convert data from stream to float array
        frame_data = librosa.util.buf_to_float(data, n_bytes=2, dtype=np.int16)

        # Create the mel spectrogram
        sgram = librosa.stft(frame_data)
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_len)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)

        # Reshape image to 4 dimensional array
        new_image = tf.reshape(mel_sgram, [1, 128, 32, 1])

        # clasify mel spec
        predictions = audio_model.predict(new_image)

        # Print exit conditions
        print(str("Press q or escape to end."))

        # print classifications
        print(str("Prediction Array: "))
        print(predictions[0])
        print(str("Prediction No: "))
        print(str(class_names[np.argmax(predictions[0])]))

        if keyboard.is_pressed("q"):
            print("q pressed")
            break
        if keyboard.is_pressed("esc"):
            print("esc pressed")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()


def open_audio_video():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('D:/Downloads/Ap_Creative_Stock_Header.jpg')
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_len,
                    input=True,
                    frames_per_buffer=frame_len)

    plt.ion()

    # Main loop for audio and video input and continuous classification
    while (True):
        ret, frame = cap.read()
        if not ret:
            continue

        img = frame
        temp_img = np.zeros([48, 48, 3], dtype=np.uint8)
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

        # make video predictions
        video_predictions = video_model.predict(resize_image)
        cv2.putText(img, str(class_names[np.argmax(video_predictions[0])]), (tx - 10, ty - 10), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 255, 0))

        # setting the scale of the output video
        scale_percent = 75
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # print classifications
        print(str("Video: "))
        print(str("Prediction Array: "))
        print(video_predictions[0])
        print(str("Prediction No: "))
        print(np.argmax(video_predictions[0]))
        print(str(class_names[np.argmax(video_predictions[0])]))

        pred = str(class_names[np.argmax(video_predictions[0])])

        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        # Audio portion
        # data read
        data = stream.read(frame_len, exception_on_overflow=False)

        # convert data from stream to float array
        frame_data = librosa.util.buf_to_float(data, n_bytes=2, dtype=np.int16)

        # Create the mel spectrogram
        sgram = librosa.stft(frame_data)
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_len)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)

        # Reshape image to 4 dimensional array
        new_image = tf.reshape(mel_sgram, [1, 128, 32, 1])

        # clasify mel spec
        audio_predictions = audio_model.predict(new_image)

        # print classifications
        print(str("Audio: "))
        print(str("Prediction Array: "))
        print(audio_predictions[0])
        print(str("Prediction No: "))
        print(np.argmax(audio_predictions[0]))
        print(str(class_names[np.argmax(audio_predictions[0])]))

        temp_predictions = audio_predictions + video_predictions

        # print combined classifications
        print(str("combined: "))
        print(str("Prediction Array: "))
        print(temp_predictions[0])
        print(str("Prediction No: "))
        print(np.argmax(temp_predictions[0]))
        print(str(class_names[np.argmax(temp_predictions[0])]))

        # add text to frame
        cv2.putText(img, str('Combined: ' + class_names[np.argmax(temp_predictions[0])]), (0, img.shape[0]),
                    cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 255, 0))

        # Display frame
        cv2.imshow("yoop", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    stream.stop_stream()
    stream.close()
    p.terminate()


# start of 'Main' program
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
