#!/bin/bash

source /opt/intel/computer_vision_sdk_2018.3.343/bin/setupvars.sh

python3 real_time_object_detection_multi.py -d movidius -i offline -s ~/workshop/computer-vision/OpenCV_Object_Detection/resources/bus_station_6094_960x540.mp4 --framework caffe --mweight ~/workshop/computer-vision/OpenCV_Object_Detection/models/MobileNetSSD.graph --model_image_height 300 --model_image_width 300 --mlabels ~/workshop/computer-vision/OpenCV_Object_Detection/models/MobileNetSSD_labels.txt -c 0.5 --infer_fc 5 --display_off true

