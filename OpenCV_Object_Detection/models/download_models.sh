#!/bin/bash

source /opt/intel/computer_vision_sdk_2018.3.343/bin/setupvars.sh

# download mobilenetssd

python3 /opt/intel/computer_vision_sdk_2018.3.343/deployment_tools/model_downloader/downloader.py --name ssd300 --output_dir .
python3 /opt/intel/computer_vision_sdk_2018.3.343/deployment_tools/model_downloader/downloader.py --name mobilenet-ssd --output_dir .

# run model optimizer

python3 /opt/intel/computer_vision_sdk_2018.3.343/deployment_tools/model_optimizer/mo.py --framework caffe --input_model ssd300.caffemodel --input_proto ssd300.prototxt
python3 /opt/intel/computer_vision_sdk_2018.3.343/deployment_tools/model_optimizer/mo.py --framework caffe --input_model mobilenet-ssd.caffemodel --input_proto mobilenet-ssd.prototxt


# download tensorflow model

wget -O ssd_mobilenet_v1_coco_2017_11_17.tar.gz http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
tar xzvf ssd_mobilenet_v1_coco_2017_11_17.tar.gz
mv ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb .
wget -O ssd_mobilenet_v1_coco_2017_11_17.pbtxt https://gist.githubusercontent.com/dkurt/45118a9c57c38677b65d6953ae62924a/raw/b0edd9e8c992c25fe1c804e77b06d20a89064871/ssd_mobilenet_v1_coco_2017_11_17.pbtxt

# download mobilenet-ssd deploy

wget -O MobileNetSSD_deploy.prototxt https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt
wget -O MobileNetSSD_deploy.caffemodel https://github.com/chuanqi305/MobileNet-SSD/blob/master/MobileNetSSD_deploy.caffemodel?raw=true
