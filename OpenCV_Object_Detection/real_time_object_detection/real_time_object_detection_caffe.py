import cv2 as cv
from imutils.video import FPS
import psutil
import os

current_process = psutil.Process(os.getpid())

confidence = 0.6

video_path = '../resources/motorcycle_6098_shortened_960x540.mp4'

#caffe model labels for objects are predefined for MobileNetSSD caffe model and protbuf
labels = ("background", "aeroplane", "bicycle",
          "bird", "boat", "bottle", "bus",
          "car", "cat", "chair", "cow",
          "diningtable", "dog", "horse",
          "motorbike", "person", "pottedplant",
          "sheep", "sofa", "train", "tvmonitor")

#label_colors = np.random.uniform(0, 255, (len(labels),3))
# we generated random colors for each label to determine object on frame
label_colors = [[ 70.03036239,  53.39948712, 221.96066983],
                   [ 72.40459246,  81.85653543, 216.79091508],
                   [ 16.05095266, 156.74660586,  28.22137944],
                   [ 22.14580474, 245.30084464, 203.24240217],
                   [ 67.87645208,  61.44175277,  62.59789847],
                   [252.36723978,   5.40010433,  73.84552673],
                   [207.95470272,  96.58437259,  17.00872131],
                   [108.9367236 , 180.97081026,  78.16660705],
                   [237.0586842 , 160.01565458, 106.49361722],
                   [131.40428931,  43.9492775 , 222.22671871],
                   [109.40802485, 123.90466382, 208.49082336],
                   [241.25056538, 246.46355905, 215.40549655],
                   [ 50.53963961, 188.7669464 ,  14.91525421],
                   [104.91164983,  13.90156432,  80.97275078],
                   [ 65.87683959, 160.34697271, 199.46650188],
                   [ 16.08423214,  84.441482  , 163.3640731 ],
                   [ 68.50589207,  65.21968418, 229.81699866],
                   [151.91579089, 195.49198107,  94.49696933],
                   [132.74947445,  14.51457431, 163.51873436],
                   [ 83.84690577, 178.29185705, 128.78807612],
                   [195.2857407 , 247.73377045, 175.55730603]]

caffe_model = '../models/MobileNetSSD_deploy.caffemodel'
caffe_proto = '../models/MobileNetSSD_deploy.prototxt'

cf_net = cv.dnn.readNetFromCaffe(caffe_proto, caffe_model)

cap = cv.VideoCapture(video_path)

fps = FPS().start()
# Process inputs
while True:

    has_frame, frame = cap.read()

    if not has_frame:
        break

    # First we get image size
    orig_rows = frame.shape[0]
    orig_cols = frame.shape[1]

    rows = 300
    cols = 300

    resized_frame = cv.resize(frame, (rows, cols))

    blob = cv.dnn.blobFromImage(resized_frame, 0.00784, (rows, cols), (127.5, 127.5, 127.5), swapRB=True, crop=False)

    cf_net.setInput(blob)

    out = cf_net.forward()

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
            #print(label_text)
            #cv.putText(resized_img, label_text, (int(left), int(top)), cv.FONT_HERSHEY_SIMPLEX, 0.5, label_colors[label_index], 2)
            #cv.rectangle(resized_img, (int(left), int(top)), (int(right), int(bottom)), label_colors[label_index], thickness=3)

            # original image
            row_factor = orig_rows / 300.0
            col_factor = orig_cols / 300.0

            # Scale object detection to original image
            left = int(col_factor * left)
            top = int(row_factor * top)
            right = int(col_factor * right)
            bottom = int(row_factor * bottom)

            cv.putText(frame, label_text, (int(left), int(top)), cv.FONT_HERSHEY_SIMPLEX, 0.5, label_colors[label_index], 2)
            cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), label_colors[label_index], thickness=2)

    cv.putText(frame, 'CPU% : {}'.format(current_process.cpu_times()[0]), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv.imshow('OpenCV and Caffe DNN', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    fps.update()

fps.stop()

print('FPS : {}'.format(round(fps.fps(), 2)))
cv.destroyAllWindows()