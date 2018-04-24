#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 18:08:23 2018

@author: serkan
"""

import re
import cv2
import numpy as np

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
    

def annotate_image(image, window, label=""):
    '''
      Generates an annotation on image given window and label
    '''
    TOP_HEIGHT = 20
    LABEL_BOTTOM_MARGIN = 5
    window_color = (23, 230, 210)
    annotation_color = (23, 230, 210, 100)
    label_color = (255,255,255)
        
    top_left = ( int(window[0]), int(window[1]) )
    bot_right = ( int(window[2]), int(window[3]) )
    

    cv2.rectangle(image, top_left, bot_right, window_color, thickness=2)   
    cv2.rectangle(image, (top_left[0], top_left[1] - TOP_HEIGHT), (bot_right[0], top_left[1]), annotation_color, -1)
    cv2.putText(image, label, (top_left[0], top_left[1] - LABEL_BOTTOM_MARGIN), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, label_color,1,cv2.LINE_AA)
