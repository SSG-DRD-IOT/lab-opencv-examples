#!/bin/bash

source /opt/intel/computer_vision_sdk_2018.3.343/bin/setupvars.sh


#display off
python3 real_time_object_detection_multi.py -d cpu -i live -s 0 --framework caffe --mconfig ~/workshop/computer-vision/OpenCV_Object_Detection/models/mobilenet-ssd.prototxt --mweight ~/workshop/computer-vision/OpenCV_Object_Detection/models/mobilenet-ssd.caffemodel --model_image_height 300 --model_image_width 300 --mlabels ~/workshop/computer-vision/OpenCV_Object_Detection/models/MobileNetSSD_labels.txt -c 0.5 


