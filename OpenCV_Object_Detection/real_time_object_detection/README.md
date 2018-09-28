## Real Time Object Detection 

## Part 0 : Deep Learning Models 

Models used in this section can be downloaded with using [../models/download_models.sh](https://github.com/SSG-DRD-IOT/lab-opencv-examples/blob/milano-workshop/OpenCV_Object_Detection/models/download_models.sh) script.

This script assumes, we alread have OpenVINO installed. 

```bash
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

```

## Part I : Inference with OpenCV using

At this tutorial, we combine the knowledge for object detection from a single image as shown in [Caffe](../OpenCV%20and%20Deep%20Neural%20Networks%20for%20Object%20Detection%20-%20Caffe.ipynb) and [Tensorflow](../OpenCV%20and%20Deep%20Neural%20Networks%20for%20Object%20Detection%20-%20Tensorflow.ipynb) examples to implement real time object detection using either a video file or web cam.

You can see the python scripts in your UP2 board from `/home/upsquared/workshop/computer-vision/OpenCV_Object_Detection/real_time_object_detection` folder.

- First script, is written with [Caffe models](real_time_object_detection_caffe.py)

- Second script, is written with [Tensorflow models](real_time_object_detection_tensorflow.py)

Main differences from the single image example:
 
 - We continuously read frames from video source.
 - Run object detection on each frame until quit command or frames end.
 - Added CPU usage to top left of window.
 
We also used additional packages to get more data about CPU and frame per second information.

Before running samples make sure you installed imutils and psutil packages

```$ pip install imutils psutil opencv-python``` 

## Part II : Inference with Intel OpenVINO

This part, we want to showcase how Intel OpenVINO helps us in computer vision, deep learning inference.

- [real_time_object_detection_multi.py](https://github.com/SSG-DRD-IOT/lab-opencv-examples/blob/milano-workshop/OpenCV_Object_Detection/real_time_object_detection/real_time_object_detection_multi.py) Python application helps us to run inference on CPU (OpenCV), GPU (OpenVINO) and VPU (Movidius - NCSDK) to make all types of comparisons.

We also inserted multiple combinations of shell script to help you run sample application.

Below script [run_openvino_object_detection_off.sh](https://github.com/SSG-DRD-IOT/lab-opencv-examples/blob/milano-workshop/OpenCV_Object_Detection/real_time_object_detection/run_openvino_object_detection_off.sh) runs application with
- MobileNetSSD Model
- Uses GPU
- Sets Inference Per Second to 5
- Switches-off the display to see results on terminal.

```bash
source /opt/intel/computer_vision_sdk_2018.3.343/bin/setupvars.sh

python3 real_time_object_detection_multi.py -d gpu -i offline -s ~/workshop/computer-vision/OpenCV_Object_Detection/resources/bus_station_6094_960x540.mp4 --mconfig ~/workshop/computer-vision/OpenCV_Object_Detection/models/mobilenet-ssd.xml --mweight ~/workshop/computer-vision/OpenCV_Object_Detection/models/mobilenet-ssd.bin --mlabels ~/workshop/computer-vision/OpenCV_Object_Detection/models/MobileNetSSD_labels.txt -c 0.5 --infer_fc 5 --display_off true

```

You can try all the .sh files to see potential results. If you are remotely connected to UP2 board and running them through SSH you should make sure that `--display_off true` set.
