#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 18:08:23 2018

@author: serkan
"""

import re
import os
import cv2
import numpy as np

__all__ = ["RNN_Detector"]

class RNN_Detector:

    def __init__(self, folder_name):
        self.model, self.label_map = RNN_Detector.read_model(folder_name)

    def generate_label_map(fname):
        '''
           Generates label map from protofile
        '''
        label_dict = {}

        with open(fname) as fp:
            text = fp.read()

            items = re.findall("item\s*{\s*[^}]*}", text)

            for item in items:
                display_name = re.findall('display_name\s*:\s*".*"', item)[0]
                display_name = re.findall('".*"', display_name)[0][1:-1]

                id = re.findall('id\s*:\s*.*', item)[0]
                id = re.findall('[0-9]+', id)[0]
                id = int(id)

                label_dict[id] = display_name
        return label_dict


    def annotate_image(self,image, window, label="", window_corner=(0,0)):
        '''
          Generates an annotation on frame given window and label
        '''
        TOP_HEIGHT = 20
        LABEL_BOTTOM_MARGIN = 5
        LABEL_HORIZONTAL_MARGIN = 1;
        window_color = (23, 230, 210)
        annotation_color = (23, 230, 210, 100)
        label_color = (255,255,255)

        top_left = ( int(window[0] + window_corner[0]), int(window[1]+ window_corner[1]) )
        bot_right = ( int(window[2]+ window_corner[0]), int(window[3]+ window_corner[1]) )

        cv2.rectangle(image, top_left, bot_right, window_color, thickness=2)
        cv2.rectangle(image, (top_left[0] - LABEL_HORIZONTAL_MARGIN, top_left[1] - TOP_HEIGHT),
                             (bot_right[0] + LABEL_HORIZONTAL_MARGIN, top_left[1]),
                             annotation_color, -1)
        cv2.putText(image, label, (top_left[0], top_left[1] - LABEL_BOTTOM_MARGIN),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, label_color,1,cv2.LINE_AA)

    def read_model(folder_name, model_file="graph.pbtxt", weight_file="frozen_inference_graph.pb",
                   label_file="labelmap.pbtxt"):
        '''
          Reads model folder and returns a detector object


          Model folder must contain:
            1) Graph model (default name: graph.pbtxt)
            2) Frozen model weights (default name: frozen_inference_graph.pb)
            3) Labelmap (default name: labelmap.pbtxt)

          Parameters:
              folder_name : Name of the model folder
          Keywords:
              model_file  : Location of the graph model (default: folder_name/graph.pbtxt)
              weight_file : Location of the weight file (default: folder_name/frozen_inference_graph.pb)
              label_file  : Location of the label file  (default: folder_name/labelmap.pbtxt)

          Returns:
              (model, label_map)
              model : Opencv model file
              label_map : A dictionary that maps label ids to label display names
        '''

        # Set default parameters
        model_file = os.path.join(folder_name, model_file)

        weight_file = os.path.join(folder_name, weight_file)

        label_file = os.path.join(folder_name, label_file)

        # Load model file
        label_map = RNN_Detector.generate_label_map(label_file)
        model = cv2.dnn.readNetFromTensorflow(weight_file, model_file)

        return model, label_map

    def detect(self, img, input_size=(300, 300), scale=1.0 / 127.5,
               mean=(127.5, 127.5, 127.5), swapRB=True, crop=False, min_score=0.3):
        '''
          Performs detection on frame using given model
            This function is a generator that returns a detection for each step

          Parameters:
              model: Model object (can be read using read_model function)
              img: Image to be used in detection
          Keywords:
              label_map: Label dictionary
                         If not provided label ids used instead
              input_size: Input size of the given model
              scale: Input scaling for model
              mean: Input mean for model
              swapRB: Whether to swap Red and Blue channels in given frame
              crop:  Whether to crop the given frame
              min_score: Minimum confidence value
          Returns:
              {
                  'window' : rendering window stored as a 4 tuple (left, top, right, bottom)
                  'label_id' : Id of the label assigned for detection
                  'label_name' : Name of the label assigned for detection (Same as label_id when label_map is not defined)
                  'score' : Confidence score of the detection
              }

        '''
        # Set input of the model as given frame
        self.model.setInput(cv2.dnn.blobFromImage(img, scale, input_size, mean, swapRB=swapRB, crop=crop))
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        out = self.model.forward()  # Perform forward step and get output

        rows, cols, _ = img.shape

        # For each detection find detection score
        #  If detection score is greater than min_score then yield detection
        for detection in out[0, 0, :, :]:
            score = float(detection[2])  # Score
            if (score >= min_score):
                label_id = int(detection[1])  # Output label

                # Find detection window
                # Window is stored as (left, top, right, bottom) format
                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows
                window = (left, top, right, bottom)

                # If label map is given translate label_id into label name
                #  Otherwise keep it same as label_id
                if self.label_map is None:
                    label_name = str(label_id)
                else:
                    label_name = self.label_map[label_id]

                # Build detection dict from calculated values
                #   Dictionary is used to make this more clear
                #   Note: Dictionary output might be slow ??
                detection_dict = {
                    'window': window,
                    'label_id': label_id,
                    'label_name': label_name,
                    'score': score
                }

                yield detection_dict
