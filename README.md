# HOG/R-CNN Pedestrian Detection & Tracking

Given an input video, detects pedestrians using either HOG (Histogram of Oriented Gradients) or R-CNN's. For performance reasons, only searches regions where movement occurs, using background subtraction.

Tracking is done using Kalman Filter. Report format is MOT compatible.
Link: https://motchallenge.net/instructions/

Uses models from: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

Requires mobilenet folder containing:

  * [frozen_inference_graph.pb](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz)
  * [graph.pbtxt](https://gist.github.com/dkurt/45118a9c57c38677b65d6953ae62924a)
  * [labelmap.pbtxt](https://raw.githubusercontent.com/tensorflow/models/ed4e22b81db3c14f48964b56580416a6936c07b0/research/object_detection/data/mscoco_label_map.pbtxt)
  
  Requires: (Tested on Python 3.7 with Anaconda)
  * OpenCV 4.2
  * Numpy 
  * Skimage 0.16.2
  * Tensorflow 2.0.0
  
