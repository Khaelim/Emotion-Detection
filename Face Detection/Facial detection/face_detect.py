import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import mtcnn
from mtcnn.mtcnn import MTCNN

# draw an image with detected objects
def draw_facebox(filename, result_list):
    # load the image
    data = plt.imread(filename)
    # plot the image
    plt.imshow(data)
    # get the context for drawing boxes
    ax = plt.gca()
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = plt.Rectangle((x, y), width, height,fill=False, color='orange')
        # draw the box
        ax.add_patch(rect)
        # draw the dots
        for key, value in result['keypoints'].items():
            # create and draw dot
            dot = plt.Circle(value, radius=20, color='red')
        ax.add_patch(dot)
    # show the plot
    plt.show()


cap = cv2.VideoCapture(0)

if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed

while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:

        filename = "temp.jpg"
        cv2.imwrite("temp.jpg", frame)
        pixels = plt.imread("temp.jpg")
        detector = mtcnn.MTCNN()
        faces = detector.detect_faces(pixels)
        draw_facebox(filename, faces)



        # Press Q on keyboard to  exit
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

# Break the loop
    else:
        break

# When everything done, release the video capture object

cap.release()

# Closes all the frames

cv2.destroyAllWindows()
