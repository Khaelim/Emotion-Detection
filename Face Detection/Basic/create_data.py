# importing required libs
import cv2, sys, numpy, os, tkinter as tk
from tkinter import simpledialog

# create a box for name input
ROOT = tk.Tk()
ROOT.withdraw()
USER_INP = simpledialog.askstring(title="Test",
                                  prompt="What's your Name?:")

# load an opencv face clasifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# haar_file = 'haarcascade_frontalface_default.xml'
# dataset of faces will be here
datasets = 'datasets'

# sub data set for my face

sub_data = USER_INP

print(sub_data)
path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.makedirs(path)

# defining sizes of images
(width, height) = (130, 100)

# face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1152)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 648)
# current files in sub_data
# print(len(os.listdir(os.path.join(datasets, sub_data))))
current = len(os.listdir(os.path.join(datasets, sub_data)))
# introduce 30 pictures of my face to the dataset
count = 1
while count < 30:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('% s/% s.png' % (path, (count + current)), face_resize)
    count += 1

    cv2.imshow('OpenCV', im)

    key = cv2.waitKey(10)
    if key == 27:
        break
# release the video capture object
webcam.release()
# destroy all windows
cv2.destroyAllWindows()
