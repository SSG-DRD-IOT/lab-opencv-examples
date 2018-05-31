## Real Time Object Detection with OpenCV, Caffe and Tensorflow

At this tutorial, we combine the knowledge for object detection from a single image as shown in [Caffe](../OpenCV%20and%20Deep%20Neural%20Networks%20for%20Object%20Detection%20-%20Caffe.ipynb) and [Tensorflow](../OpenCV%20and%20Deep%20Neural%20Networks%20for%20Object%20Detection%20-%20Tensorflow.ipynb) examples to implement real time object detection using either a video file or web cam.

First script, is written with [Caffe models](real_time_object_detection_caffe.py)

Second script, is written with [Tensorflow models](real_time_object_detection_tensorflow.py)

Main differences from the single image example:
 
 - We continuously read frames from video source.
 - Run object detection on each frame until quit command or frames end.
 - Added CPU usage to top left of window.
 
We also used additional packages to get more data about CPU and frame per second information.

Before running samples make sure you installed imutils and psutil packages

```$ pip install imutils psutil opencv-python``` 



