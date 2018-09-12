#!/bin/bash

source /opt/intel/computer_vision_sdk_2018.3.343/bin/setupvars.sh

python3 real_time_object_detection_multi.py -d gpu -i offline -s ~/workshop/computer-vision/OpenCV_Object_Detection/resources/bus_station_6094_960x540.mp4 --mconfig ~/workshop/computer-vision/OpenCV_Object_Detection/models/ssd300.xml --mweight ~/workshop/computer-vision/OpenCV_Object_Detection/models/ssd300.bin --mlabels ~/workshop/computer-vision/OpenCV_Object_Detection/models/MobileNetSSD_labels.txt -c 0.7 --infer_fc 2

