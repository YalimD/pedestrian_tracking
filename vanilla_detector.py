#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:50:19 2018

@author: serkan
@coauthor: yalim
TODO: Check gluoncv for detection and segmentation
"""

import os
import cv2
import numpy as np
from detection_tracking_lib import *
import argparse
from datetime import datetime


if __name__ == "__main__":
    WINDOW_NAME = 'Detection Output'

    # Parse inputs
    parser = argparse.ArgumentParser(
        description="Runs model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-s', '--source', help="Source of the video", default='0')
    parser.add_argument('-c', '--confidence', help="Detection confidence", type=float, default=0.3)
    parser.add_argument('-d', '--detector', help="The detector to be used (if rnn, pass the folder containing the related"
                                                 "graph files)",default="mobilenet")
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
    ped_detector = rnn_detection.RNN_Detector(args.detector)

    # Initialize video capture
    cap = cv2.VideoCapture(source)

    prev_time = 0
    mean_fps = 0
    while cap.isOpened():
        # Time calculation
        cur_time = datetime.now().microsecond
        time_elapsed = cur_time - prev_time  # In seconds
        cur_fps = 1000000.0 / time_elapsed
        mean_fps = mean_fps * 0.5 + cur_fps * 0.5
        prev_time = cur_time

        success, image = cap.read()  # Get frame from video capture

        # Bail out if we cannot read webcam
        if success == False:
            print('Cannot read frame from source')
            break

        # For each detection
        for detection in ped_detector.detect(image, min_score=args.confidence):
            annotation_label = '{} {:.2f}%'.format(detection['label_name'], detection['score'] * 100)
            ped_detector.annotate_image(image, detection['window'], label=annotation_label)

        # print('Press esc or Q to stop')
        cv2.putText(image, "FPS: {}".format(mean_fps), (10, 15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (100, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, image)
        key = cv2.waitKey(5)

        # Bail out if q / Q / ESC is pressed
        if key == ord('q') or key == ord('Q') or key == 23:
            break

        # Bail out if window is closed
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break

    cap.release()