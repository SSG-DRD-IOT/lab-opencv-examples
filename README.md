## OpenCV Examples 

At this part of the excercises for Industrial IoT, we will try to cover the basic principles of visual computing using OpenCV, and some practices to use OpenCV Python library with Python Notebooks.

Many times instead of raw presentations, better to start experiments directly hands on and learn.

As per Industrial IoT Workshops, UP2 boards are already Jupyter notebook servers, so you need to connect to UP2 board with its IP address from your browser:
```
http:\\<up2_board_ip_address>:8888
```
When prompted as below image, write **upsquared** for password.

And navigate to: ```/home/upsquared/Desktop/Lab\ Answers/lab-opencv-examples```

Your image might be out dated so clone this repository to your choice of folder after connecting with ssh.

```bash
$ git clone https://github.com/SSG-DRD-IOT/lab-opencv-examples.git --branch milano-workshop
```

Now, let's follow below examples to learn more about visual computing with OpenCV.

### Lab 0 - Installing 

At this step, we are trying to install required libraries.

See IPython Notebook: [Install OpenCV Versions](https://github.com/SSG-DRD-IOT/lab-opencv-examples/blob/milano-workshop/OpenCV_Excersizes/Install%20OpenCV%20Versions.ipynb)

### Lab 1 - OpenCV Basics

At this step, we try to show basics of using OpenCV Python library

See IPython Notebook: [OpenCV Basics](https://github.com/SSG-DRD-IOT/lab-opencv-examples/blob/milano-workshop/OpenCV_Excersizes/OpenCV_Basics.ipynb)

### Lab 2 - Edge Detection

At this lab, we try to show how edges detected with OpenCV Python library

See IPython Notebook: [OpenCV Edge Detection](https://github.com/SSG-DRD-IOT/lab-opencv-examples/blob/milano-workshop/OpenCV_Excersizes/OpenCV%20Edge%20Detection.ipynb)

### Lab 3 - Face Detection 

At this lab, we try to show the face detection using OpenCV Python library

See IPython Notebook: [OpenCV Face Detection](https://github.com/SSG-DRD-IOT/lab-opencv-examples/blob/milano-workshop/OpenCV_Excersizes/OpenCV%20Face%20Detection.ipynb)

### Lab 4 - Road Lane Detection

At this lab, we try to cover a common problem for autonomous driving and try to teach how OpenCV helps over lane detection on the road.

See sub folder for [Lane Detection](https://github.com/SSG-DRD-IOT/lab-opencv-examples/tree/milano-workshop/Road_Lane_Detection) and continue tutorial from there.

### Lab 5 - Object Detection using DNN (Caffe, Tensorflow) 

At this lab, we will use existing caffe and tensorflow neural network models to predict or detect objects inside images.

- [Caffe Framework Example](https://github.com/SSG-DRD-IOT/lab-opencv-examples/blob/milano-workshop/OpenCV_Object_Detection/OpenCV%20and%20Deep%20Neural%20Networks%20for%20Object%20Detection%20-%20Caffe.ipynb)

- [Tensor Framework Example](https://github.com/SSG-DRD-IOT/lab-opencv-examples/blob/milano-workshop/OpenCV_Object_Detection/OpenCV%20and%20Deep%20Neural%20Networks%20for%20Object%20Detection%20-%20Tensorflow.ipynb) 

### Lab 6 - Realtime Object Detection with Tensorflow and Caffe

At this section, we go one more step to use previously learned skills to detect objects in images for real time detection. Therefore, we utilize Tensorflow and Caffe for realtime object detection.

[Real Time Object Detection with Caffe and Tensorflow](https://github.com/SSG-DRD-IOT/lab-opencv-examples/tree/milano-workshop/OpenCV_Object_Detection/real_time_object_detection)
