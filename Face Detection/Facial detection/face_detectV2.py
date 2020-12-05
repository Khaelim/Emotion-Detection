import cv2
import matplotlib.pyplot as plt
from mtcnn import mtcnn
from facenet_pytorch import MTCNN
from PIL import Image
import torch
from imutils.video import FileVideoStream
import time
import glob
from tqdm.notebook import tqdm

from FastMTCNN import FastMTCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

filenames = ["glediston-bastos-ZtmmR9D_2tA-unsplash.jpg","glediston-bastos-ZtmmR9D_2tA-unsplash.jpg"]

# define our extractor
fast_mtcnn = FastMTCNN(
stride=4,
resize=0.5,
margin=14,
factor=0.6,
keep_all=True,
device=device
)


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

def run_detection(fast_mtcnn, filenames):
    frames = []
    frames_processed = 0
    faces_detected = 0
    batch_size = 60
    start = time.time()
    for filename in tqdm(filenames):
        v_cap = FileVideoStream(filename).start()
        v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))

        for j in range(v_len):
            frame = v_cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            if len(frames) >= batch_size or j == v_len - 1:
                faces = fast_mtcnn(frames)

                frames_processed += len(frames)
                faces_detected += len(faces)
                frames = []

                print(
                    f'Frames per second: {frames_processed / (time.time() - start):.3f},',
                    f'faces detected: {faces_detected}\r',
                    end=''
                )

                v_cap.stop()

                run_detection(fast_mtcnn, filenames)


cap = cv2.VideoCapture('Soul-Mates.AU.S01E01.WEB-DLx264-JIVE.mp4')

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
