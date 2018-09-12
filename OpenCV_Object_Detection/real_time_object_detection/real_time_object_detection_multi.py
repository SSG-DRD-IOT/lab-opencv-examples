# MIT License
#
# Copyright (c) 2017 Intel SSG DRD IOT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys, os, time

#import MVNC for Movidius Support
sys.path.insert(0, "/home/upsquared/movidius/ncappzoo/ncapi2_shim")
sys.path.insert(0, "/home/upsquared/movidius/ncsdk-2.05.00.02/api/python")
import mvnc_simple_api as mvnc

import cv2 as cv
import numpy as np
import psutil

sys.path.insert(0, "/opt/intel/computer_vision_sdk_2018.3.343/python/python3.5/ubuntu16")
sys.path.insert(0, "/opt/intel/computer_vision_sdk_2018.3.343/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64")
from openvino.inference_engine import IENetwork, IEPlugin

# get process module
process = psutil.Process(os.getpid())


class Config():
    """
    Config Model to Store Arguments
    """
    confidence = 0.6  # object recognition confidence
    source = str()  # source, live or offline
    framework = str()  # caffe or tensorflow
    video_path = str()  # full path or 0 for live
    platform = str()  # cpu, gpu or movidius
    model_labels = list()  # list of strings
    label_colors = list()  # colors to draw at post process
    model_labels_file = str()  #labels file
    model_file = str()  # model file
    model_weight_file = str()  # model weights file, if movidius graph file
    infer_frame_rate = -1  # frame rate to infer, if -1 then all frames will be inferred
    model_image_height = 300
    model_image_width = 300
    fps_delay = 1 # used when reading from a video file, according to system speed it can be fast
    display_off = False


def parse_labels(label_file_name=str()):
    """
    Parse Labels file
    Assumes each line has only one label no other separators etc.
    Convert file to this format before running
    :param label_file_name:
    :return:
    """
    if not os.path.isfile(label_file_name):
        print('Label File Not Found')
        print_help_menu()
        sys.exit(2)

    label_list = list()
    with open(label_file_name) as f:
        lines = f.readlines()
        for line in lines:
            label = line.replace('\n', '')
            label_list.append(label)

    return label_list


def get_label_colors(labels=list()):
    """
    Generate random RGB values for label colors to post process frames for drawing.
    :param labels:
    :return:
    """
    return np.random.uniform(0, 255, (len(labels), 3))


def print_help_menu():
    """
    Print help menu
    :return:
    """
    print('This script helps you to run real time object detection sample using OpenCV and NCSDK APIs using Tensorflow or Caffe DNN Models')

    print('./RealTimeObjectDetection [option] [value]' )

    print('options:')
    print('--help print this menu')
    print('-d, --device cpu|movidius|gpu : hw platform')
    print('-i, --input live|offline : source of video, either webcam or video on disk')
    print('-s, --source <full name to video> : video file full e.g. /home/videos/test.mp4')
    print('-f, --framework caffe|tensorflow : framework of models being used')
    print('--mconfig <full name of caffe prototxt or tensoflow pbtxt> file')
    print('--mweight <full name of caffe caffemodel or tensoflow pb> file, if movidius selected .graph file path should be here')
    print('--mlabels <full name of labels file, each line will have one label>')
    print('--model_image_height DNN model image height to be used for inferrence')
    print('--model_image_width DNN model image width to be used for inferrence')
    print('-c, --confidence confidence value, default 0.6')
    print('--infer_fc <1, 2 ..> number of frames to infer, by default program tries to infer as much as it can')
    print('--display-off True|False doesnt show output if enabled, default False')

    return None


def parse_args(argv):
    """
    Parse command line arguments using getopt
    :param argv:
    :return:
    """

    opts = list()

    for i in range(0, len(argv), 2):
        if argv[0] in ('-h','--help'):
            print_help_menu()
            sys.exit(0)
        else:
            opts.append((argv[i], argv[i+1]))

    for opt, arg in opts:
        if opt in ('-i', '--input'):
            Config.source = arg
        elif opt in ('-s', '--source'):
            Config.video_path = arg
        elif opt in ('-f', '--framework'):
            Config.framework = arg
        elif opt in ('-d', '--device'):
            Config.platform = arg
        elif opt in ('-c', '--confidence'):
            Config.confidence = float(arg)
        elif opt == '--mconfig':
            Config.model_file = arg
        elif opt == '--mweight':
            Config.model_weight_file = arg
        elif opt == '--mlabels':
            Config.model_labels_file = arg
        elif opt == '--infer_fc':
            Config.infer_frame_rate = int(arg)
        elif opt == '--model_image_height':
            Config.model_image_height = int(arg)
        elif opt == '--model_image_width':
            Config.model_image_width = int(arg)
        elif opt == '--display_off':
            Config.display_off = True
        else:
            print('Unknown argument exiting ...')
            sys.exit(2)

    return None


def print_summary():
    """
    Print summary of parameters
    :return:
    """
    print("Video Source {}".format(Config.source))
    print("Video Path {}".format(Config.video_path))
    print("Model Framework {}".format(Config.framework))
    print("Inference Platform {}".format(Config.platform))
    print("Model config file {}".format(Config.model_file))
    print("Model weights file {}".format(Config.model_weight_file))
    print("Model labels file {}".format(Config.model_labels_file))
    print("Model inference frames {}".format(Config.infer_frame_rate))
    print("Model input image height {}".format(Config.model_image_height))
    print("Model input image width {}".format(Config.model_image_width))

    return None


def get_video(source, path):
    """
    Load Video Source.
    Decode Video Stream
    :param source:
    :param path:
    :return:
    """
    if source == 'live':
        cap = cv.VideoCapture(0)
        #cap.set(cv.CAP_PROP_FPS, 48)
        print('Video FPS: {}'.format(cap.get(cv.CAP_PROP_FPS)))
        Config.fps_delay = int(1000 / cap.get(cv.CAP_PROP_FPS))
        return cap
    elif source == 'offline':
        if not os.path.isfile(path):
            print('Video File Not Found')
            sys.exit(2)

        cap = cv.VideoCapture(path)
        print('Video FPS: {}'.format(cap.get(cv.CAP_PROP_FPS)))
        Config.fps_delay = int(1000 / cap.get(cv.CAP_PROP_FPS))
        return cap
    else:
        print("Wrong source definition")
        print('-i, --input live|offline : source of video, either webcam or video on disk')
        sys.exit(2)


def infer_with_movidius(frame, movidius_graph):
    """
    Run inference on movidius
    :param image: resized image
    :return: list [ [ label_id, x1, y1, x2, y2, confidence], ... ]
    """
    # trasnform values from range 0-255 to range -1.0 - 1.0q
    frame = frame - 127.5
    frame = frame * 0.007843

    movidius_graph.LoadTensor(frame.astype(np.float16), None)

    output, userobj = movidius_graph.GetResult()

    return output


def infer_with_cpu(frame, network):
    """
    Run inference using opencv dnn interface.
    :param image: resized frame
    :return:
    """

    # MobileNetSSD Expects 300x300 resized frames
    blob = cv.dnn.blobFromImage(frame, 0.00784, (Config.model_image_height, Config.model_image_width), (127.5, 127.5, 127.5), swapRB=True, crop=False)

    # Send blob data to Network
    network.setInput(blob)

    # Make network do a forward propagation to get recognition matrix
    out = network.forward()

    return out[0, 0, :, :]


def infer_with_openvino(frame, input_blob, out_blob, exec_net):

    res = exec_net.infer(inputs={input_blob: frame})

    res = res[out_blob]

    return res


def post_process(original, detections):
    """
    Post process, draw rectangles
    :param original:
    :param detections:
    :return:
    """
    if Config.platform == 'movidius':
        num_valid_boxes = int(detections[0])
        # print('Number of Valid Detections {}'.format(num_valid_boxes))

        for box_index in range(num_valid_boxes):
            base_index = 7 + box_index * 7
            if (not np.isfinite(detections[base_index]) or
                    not np.isfinite(detections[base_index + 1]) or
                    not np.isfinite(detections[base_index + 2]) or
                    not np.isfinite(detections[base_index + 3]) or
                    not np.isfinite(detections[base_index + 4]) or
                    not np.isfinite(detections[base_index + 5]) or
                    not np.isfinite(detections[base_index + 6])):
                # boxes with non finite (inf, nan, etc) numbers must be ignored
                continue

            left = max(int(detections[base_index + 3] * Config.model_image_width), 0)
            top = max(int(detections[base_index + 4] * Config.model_image_height), 0)
            right = min(int(detections[base_index + 5] * Config.model_image_width), 300 - 1)
            bottom = min((detections[base_index + 6] * Config.model_image_height), Config.model_image_height - 1)

            object_info = detections[base_index:base_index + 7]

            base_index = 0

            label_index = int(object_info[base_index + 1])

            if label_index < 0:
                continue

            score = object_info[base_index + 2]

            if score > Config.confidence:
                label_text = Config.model_labels[label_index] + " " + str(round(score, 4))

                # original image
                row_factor = original.shape[0] / float(Config.model_image_height)
                col_factor = original.shape[1] / float(Config.model_image_width)

                # Scale object detection to original image
                left = int(col_factor * left)
                top = int(row_factor * top)
                right = int(col_factor * right)
                bottom = int(row_factor * bottom)

                if not Config.display_off:
                     cv.putText(original, label_text, (int(left), int(top)), cv.FONT_HERSHEY_SIMPLEX, 0.5, Config.label_colors[label_index], 2)
                     cv.rectangle(original, (int(left), int(top)), (int(right), int(bottom)), Config.label_colors[label_index], thickness=3)

                print('Detected {} with {} Confidence'.format(label_text, score))
                print('CPU Count: {} - CPU% : {} '.format(psutil.cpu_count(), process.cpu_percent()))

                if not Config.infer_frame_rate == -1:
                    print('Inference Frame Rate: {}'.format(Config.infer_frame_rate))

    elif Config.platform == 'cpu':
        # go over objects
        for detection in detections:
            # confidence score
            score = float(detection[2])

            # label index
            label_index = int(detection[1])

            # draw rectangle and write the name of the object if above given confidence
            if score > Config.confidence:
                left = detection[3] * Config.model_image_width
                top = detection[4] * Config.model_image_height
                right = detection[5] * Config.model_image_width
                bottom = detection[6] * Config.model_image_height

                label_text = Config.model_labels[label_index] + " " + str(round(score, 4))

                # original image
                row_factor = original.shape[0] / float(Config.model_image_height)
                col_factor = original.shape[1] / float(Config.model_image_width)

                # Scale object detection to original image
                left = int(col_factor * left)
                top = int(row_factor * top)
                right = int(col_factor * right)
                bottom = int(row_factor * bottom)

                if not Config.display_off:
                    cv.putText(original, label_text, (int(left), int(top)), cv.FONT_HERSHEY_SIMPLEX, 0.5, Config.label_colors[label_index], 2)
                    cv.rectangle(original, (int(left), int(top)), (int(right), int(bottom)), Config.label_colors[label_index], thickness=3)

                print('Detected {} with {} Confidence'.format(label_text, score))
                print('CPU Count: {} - CPU% : {} '.format(psutil.cpu_count(), process.cpu_percent()))

                if not Config.infer_frame_rate == -1:
                    print('Inference Frame Rate: {}'.format(Config.infer_frame_rate))

    elif Config.platform == 'gpu':
        #print(detections)
        for detection in detections[0][0]:
            #print(detection)
            # confidence score
            score = float(detection[2])

            # label index
            label_index = int(detection[1])

            # draw rectangle and write the name of the object if above given confidence
            if score > Config.confidence:
                left = detection[3] * Config.model_image_width
                top = detection[4] * Config.model_image_height
                right = detection[5] * Config.model_image_width
                bottom = detection[6] * Config.model_image_height

                label_text = Config.model_labels[label_index] + " " + str(round(score, 4))

                # original image
                row_factor = original.shape[0] / float(Config.model_image_height)
                col_factor = original.shape[1] / float(Config.model_image_width)

                # Scale object detection to original image
                left = int(col_factor * left)
                top = int(row_factor * top)
                right = int(col_factor * right)
                bottom = int(row_factor * bottom)

                if not Config.display_off:
                    cv.putText(original, label_text, (int(left), int(top)), cv.FONT_HERSHEY_SIMPLEX, 0.5, Config.label_colors[label_index], 2)
                    cv.rectangle(original, (int(left), int(top)), (int(right), int(bottom)), Config.label_colors[label_index], thickness=3)

                print('Detected {} with {} Confidence'.format(label_text, score))
                print('CPU Count: {} - CPU% : {} '.format(psutil.cpu_count(), process.cpu_percent()))
                
                if not Config.infer_frame_rate == -1:
                    print('Inference Frame Rate: {}'.format(Config.infer_frame_rate))

    return original


def main(argv):

    parse_args(argv)

    Config.model_labels = parse_labels(Config.model_labels_file)
    Config.label_colors = get_label_colors(Config.model_labels)

    print_summary()

    # open video source
    cap = get_video(Config.source, Config.video_path)

    print("Video Properties:")
    print("Loaded {} video from source {}".format(Config.source, Config.video_path))

    # get frame width/height
    actual_frame_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    actual_frame_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    print("Video Resolution {} x {}".format(actual_frame_width, actual_frame_height))

    # load NN model for platform
    if Config.platform == 'cpu':
        if Config.framework == 'caffe':
            net = cv.dnn.readNetFromCaffe(Config.model_file, Config.model_weight_file)
        elif Config.framework == 'tensorflow':
            net = cv.dnn.readNetFromTensorflow(Config.model_file, Config.model_weight_file)
        else:
            print("{} Framework not supported".format(Config.framework))
            sys.exit(2)

    elif Config.platform == 'movidius':
        mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)
        # Get a list of ALL the sticks that are plugged in
        mv_devices = mvnc.EnumerateDevices()
        if len(mv_devices) == 0:
            print('No Movidius devices found')
            sys.exit(2)
        else:
            print('{} Movidius Device/s Found'.format(len(mv_devices)))

        # Pick the first stick to run the network
        movidius = mvnc.Device(mv_devices[0])
        movidius.OpenDevice()

        with open(Config.model_weight_file, mode='rb') as f:
            graph_data = f.read()

        # allocate the Graph instance from NCAPI by passing the memory buffer
        movidius_graph = movidius.AllocateGraph(graph_data)

    elif Config.platform == 'gpu':
        plugin = IEPlugin(device='GPU', plugin_dirs='/opt/intel/computer_vision_sdk_2018.3.343/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64')
        print("Reading IR...")
        net = IENetwork.from_ir(model=Config.model_file, weights=Config.model_weight_file)

        input_blob = next(iter(net.inputs))
        out_blob = next(iter(net.outputs))

        n, c, h, w = net.inputs[input_blob]

        print("Loading IR to the plugin...")
        exec_net = plugin.load(network=net)
    else:
        print("{} Platform not supported".format(Config.platform))
        print('-d, --device cpu|movidius|gpu : hw platform')
        sys.exit(2)

    # Start Counting frames to Calculate FPS
    frame_count = 0
    start_time = time.time()
    inferred_frame_count = 0

    if not Config.display_off:
        cv.namedWindow("Real Time Object Detection", cv.WINDOW_FULLSCREEN)

    detections = None
    cur_request_id = 0

    print("Starting inference with {}...".format(Config.platform))
    print("Q to Quit")

    while True:
        # read frame from capture
        has_frame, frame = cap.read()
        infer_start_time = time.time()

        if not has_frame:
            end_time = time.time()
            print("No more frame from from video source, exiting ....")
            break

        if Config.infer_frame_rate == -1 or inferred_frame_count / (time.time() - start_time) < Config.infer_frame_rate:
            if Config.platform == 'cpu' or Config.platform == 'movidius':
                resized_frame = cv.resize(frame, (Config.model_image_height, Config.model_image_width))
            else:
                resized_frame = cv.resize(frame, (w, h))

            if Config.platform == 'cpu':
                detections = infer_with_cpu(resized_frame, net)
            elif Config.platform == 'movidius':
                detections = infer_with_movidius(resized_frame, movidius_graph)
            elif Config.platform == 'gpu':
                resized_frame = resized_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                resized_frame = resized_frame.reshape((n, c, h, w))
                detections = infer_with_openvino(resized_frame, input_blob, out_blob, exec_net)
            else:
                print('Platform not found')
                sys.exit(2)

            inferred_frame_count += 1

        infer_end_time = time.time()

        print('Elapsed Inference Time: {} ms'.format((infer_end_time - infer_start_time)*1000.))

        if detections is not None:
            frame = post_process(frame, detections)

        if Config.infer_frame_rate == -1:
            ifr = float(inferred_frame_count) / (start_time - time.time())

        # display text to let user know how to quit
        if not Config.display_off:
            cv.rectangle(frame, (0, 0), (220, 60), (50, 50, 50, 100), -1)
            cv.putText(frame, "Q to Quit", (10, 12), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv.putText(frame, 'CPU Count: {} - CPU% : {} '.format(psutil.cpu_count(), process.cpu_percent()), (10, 35), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
            if Config.infer_frame_rate == -1:
                cv.putText(frame, 'Inference Frame Rate: {}'.format(ifr), (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                cv.putText(frame, 'Inference Frame Rate: {}'.format(Config.infer_frame_rate), (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv.imshow('Real Time Object Detection', frame)

        frame_count += 1

        if cv.waitKey(Config.fps_delay) & 0xFF == ord('q'):
            end_time = time.time()
            break
        #TODO: UP Key to increase IPF , DOWN Key to decrease IPF

    frames_per_second = frame_count / (end_time - start_time)
    print('Calculated Frames Per Second: ' + str(frames_per_second))

    if not Config.display_off:
         cv.destroyAllWindows()

    # Release Movidius Device Allocation
    if Config.platform == 'movidius':
        movidius_graph.DeallocateGraph()
        movidius.CloseDevice()

    if Config.platform == 'gpu':
        del exec_net
        del plugin

    cap.release()


if __name__ == "__main__":
    main(sys.argv[1:])
