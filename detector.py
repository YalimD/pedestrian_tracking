#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:50:19 2018

@author: serkan
@coauthor: yalim
"""

import os
import cv2
import numpy as np
from util import generate_label_map, annotate_image
import argparse
from datetime import datetime 
        
def read_model(folder_name, model_file="graph.pbtxt", weight_file="frozen_inference_graph.pb", label_file="labelmap.pbtxt"):
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
    label_map = generate_label_map(label_file)
    model = cv2.dnn.readNetFromTensorflow(weight_file, model_file)
    
    return model, label_map
    
def detect(model, img, label_map=None, input_size=(300, 300), scale=1.0/127.5,
           mean=(127.5, 127.5, 127.5), swapRB=True, crop=False, min_score=0.3):
    '''
      Performs detection on image using given model
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
          swapRB: Whether to swap Red and Blue channels in given image
          crop:  Whether to crop the given image
          min_score: Minimum confidence value
      Returns:
          {
              'window' : rendering window stored as a 4 tuple (left, top, right, bottom)
              'label_id' : Id of the label assigned for detection
              'label_name' : Name of the label assigned for detection (Same as label_id when label_map is not defined)
              'score' : Confidence score of the detection
          }
          
    '''
    # Set input of the model as given image
    model.setInput(cv2.dnn.blobFromImage(img, scale, input_size, mean, swapRB=swapRB, crop=crop))
    out = model.forward() # Perform forward step and get output 

    rows, cols, _ = img.shape
    
    # For each detection find detection score
    #  If detection score is greater than min_score then yield detection
    for detection in out[0,0,:,:]:
        score = float(detection[2]) # Score
        if (score >= min_score):
            label_id = int(detection[1]) # Output label
            
            #Find detection window
            # Window is stored as (left, top, right, bottom) format
            left = detection[3] * cols 
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            window = (left, top, right, bottom)
            
            # If label map is given translate label_id into label name
            #  Otherwise keep it same as label_id
            if label_map is None:
                label_name = str(label_id)
            else:
                label_name = label_map[label_id]
                
            # Build detection dict from calculated values
            #   Dictionary is used to make this more clear
            #   Note: Dictionary output might be slow ??
            detection_dict = {
                    'window' : window,
                    'label_id' : label_id,
                    'label_name' : label_name,
                    'score' : score
            }
            
            yield detection_dict
    
if __name__ == "__main__":
    WINDOW_NAME = 'Detection Output'
    
    # Parse inputs
    parser = argparse.ArgumentParser(
            description="Runs model",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument('-s','--source', help="Source of the video", default='0')
    parser.add_argument('-c','--confidence', help="Detection confidence", type=float, default=0.3)
    args = parser.parse_args()
    
    if (args.source and args.source != '0'):
        print("Playing from source {}".format(args.source))
        source = args.source
        if (source.isdigit()):
            source = int(source)
    else:
        source = 0
        print("Playing from default source")
        
    # Load model
    model, label_map = read_model('mobilenet')
    
    # Initialize video capture
    cap = cv2.VideoCapture(source)

    prev_time = datetime.now().microsecond
    mean_fps = 0
    while cap.isOpened():
        # Time calculation
        cur_time = datetime.now().microsecond
        time_elapsed = cur_time - prev_time	# In seconds
        cur_fps = 1000000.0 / time_elapsed
        mean_fps = mean_fps * 0.5 + cur_fps * 0.5
        prev_time = cur_time  

        success, image = cap.read() # Get image from video capture

        # Bail out if we cannot read webcam
        if success == False:
            print('Cannot read image from source')
            break

        # For each detection
        for detection in detect(model, image, label_map = label_map, min_score=args.confidence):
            annotation_label = '{} {:.2f}%'.format(detection['label_name'], detection['score'] * 100)
            annotate_image(image, detection['window'], label=annotation_label)

        print('Press esc or Q to stop')

        cv2.imshow(WINDOW_NAME,image)
        key = cv2.waitKey(5)
        
        # Bail out if q / Q / ESC is pressed
        if key == ord('q') or key == ord('Q') or key == 23:
            break
        
        # Bail out if window is closed
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break

    cap.release()
