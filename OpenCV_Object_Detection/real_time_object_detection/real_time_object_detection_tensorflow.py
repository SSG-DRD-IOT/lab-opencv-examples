import cv2 as cv
import numpy as np
from imutils.video import FPS
import psutil
import os

current_process = psutil.Process(os.getpid())

confidence = 0.6

video_path = '../resources/bus_station_6094_960x540.mp4'

# coco labels
# there are 183 labels, all can be find in models/mscoco_labels.txt, below code parses labels to reuse them
tf_labels_file = '../models/mscoco_labels.txt'

# read labels
labels = list()

with open(tf_labels_file) as f:
    lines = f.readlines()
    for line in lines:
        items = line.split(': ')
        labels.append(items[1].replace('\n', ''))

label_colors = np.random.uniform(0, 255, (len(labels), 3))

# here is the models
tf_model_file = '../models/frozen_inference_graph.pb'
tf_config_file = '../models/ssd_mobilenet_v1_coco_2017_11_17.pbtxt'

# Load Tensorflow models
tf_net = cv.dnn.readNetFromTensorflow(tf_model_file, tf_config_file)

# open offline video source
cap = cv.VideoCapture(video_path)

# open video from camera
# cap = cv.VideoCapture(0)

# use FPS to calculate processes frames per second
# https://github.com/jrosebr1/imutils/blob/master/imutils/video/fps.py

fps = FPS().start()

while True:

    hasFrame, frame = cap.read()

    if not hasFrame:
        break

    # First we get image size
    rows = frame.shape[0]
    cols = frame.shape[1]

    blob = cv.dnn.blobFromImage(frame, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)

    tf_net.setInput(blob)

    out = tf_net.forward()

    for detection in out[0, 0, :, :]:
        # confidence score
        score = float(detection[2])

        # label index
        label_index = int(detection[1])

        # draw rectangle and write the name of the object if above given confidence
        if score > confidence:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows

            label_text = labels[label_index] + " " + str(round(score, 4))

            cv.putText(frame, label_text, (int(left), int(top)), cv.FONT_HERSHEY_SIMPLEX, 0.5, label_colors[label_index], 2)
            cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), label_colors[label_index], thickness=3)

    cv.putText(frame, 'CPU Count: {} - CPU% : {}'.format(psutil.cpu_count(), current_process.cpu_percent()), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv.imshow('OpenCV and Tensorflow DNN', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    fps.update()

fps.stop()

print('FPS : {}'.format(round(fps.fps(), 2)))
cv.destroyAllWindows()
