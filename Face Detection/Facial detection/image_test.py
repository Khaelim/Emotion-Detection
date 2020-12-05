import matplotlib.pyplot as plt
#from facenet_pytorch import MTCNN
from facenet_pytorch.models import mtcnn
from mtcnn.mtcnn import MTCNN
from PIL import Image
import torch
from imutils.video import FileVideoStream
import cv2
import time
import glob
from tqdm.notebook import tqdm

from FastMTCNN import FastMTCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

fast_mtcnn = FastMTCNN(
    stride=4,
    resize=1,
    margin=14,
    factor=0.6,
    keep_all=True,
    device=device
)

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

def draw_facebox(filename, result_list):
    data = plt.imread(filename)
    plt.imshow(data)
    ax = plt.gca()

    #print(result_list.__call__())

    # for result in result_list:
    #     print(result)
    #
    #     x, y, width, height = result['box']
    #     rect = plt.Rectangle((x, y), width, height, fill=False, color='green')
    #     ax.add_patch(rect)

    plt.show()

#imageboi = cv2.imread('12.png')
cap = cv2.VideoCapture(0)

#

while (cap.isOpened()):
    ret, frame = cap.read()
    detector = FastMTCNN(50)

    if not ret:
        continue

    cv2.imshow("HERE", frame)
    face = FastMTCNN(frame)

    # if face[0] is None:
    #     print("Nope")
    # else:
    #     print("yup")

    #filename = "temp.jpg"
    #cv2.imwrite(filename, frame)




    #if face:
        #run_detection(FastMTCNN, '12.png')
        #draw_facebox('12.png', face)




    # Press Q on keyboard to  exit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    # Break the loop
    # else:
    #     break
cv2.destroyAllWindows()
