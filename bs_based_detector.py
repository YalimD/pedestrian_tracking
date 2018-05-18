#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:50:19 2018

@author: yalim
@coauthor: serkan
"""

import os
import cv2
import argparse
import numpy as np

from skimage import measure
from datetime import datetime
from detection_lib import *

#NOTES:
#As it appears, when the pedestrian is occluded (by a tree) the detection window includes the
#large portion of the tree. In our case, it even includes the pedestrian's shadow
#30 confidence includes shadows and some uncomprehensible images
#33 seems like a sweet spot

# TODO: (2)Create a background subtractor (which is going to be a generator class)
# that finds the possible locations of the pedestrians
# and search that area for the pedestrian itself, which excludes its shadow (hopefully)

if __name__ == "__main__":
    WINDOW_NAME = 'Detection Output'

    # Parse inputs
    parser = argparse.ArgumentParser(
        description="Runs model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-s', '--source', help="Source of the video", default='0')
    parser.add_argument('-c', '--confidence', help="Detection confidence", type=float, default=0.3)
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

    # Initialize the background subtractor
    # CNT is too noisy but can be cleared using erosion
    # GSOC has a lot of initial noise, but gets better in time
    # LSBP has a lot of noise
    # MOG is improved compared to the previous version
    backgroundSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

    prev_time = 0
    mean_fps = 0
    frameNum = 0
    while cap.isOpened():
        # Time calculation
        cur_time = datetime.now().microsecond
        time_elapsed = cur_time - prev_time  # In seconds
        cur_fps = 1000000.0 / time_elapsed
        mean_fps = mean_fps * 0.5 + cur_fps * 0.5
        prev_time = cur_time

        success, image = cap.read()  # Get image from video capture

        # Bail out if we cannot read webcam
        if success == False:
            print('Cannot read image from source')
            break

        #TODO: (3) Test on zoomed in (increased size) image
        foreground_mask = backgroundSubtractor.apply(image, 0.9)

        # Now search for pedestrians in each contour in the background subtracted image
        # Each contours's bounding box is searched

        # Apply dilation, erosion

        closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, closing_kernel, foreground_mask)
        cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, opening_kernel, foreground_mask)

        temp_image, contours, hierarchy = cv2.findContours(foreground_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_GRAY2BGR)


        boxes = []
        rects = []
        # Define bounding boxes of the contours
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])

            # The rotated rectangle (Not necessary)
            # rect = cv2.minAreaRect(contour)
            # rects.append(rect)
            # box = cv2.boxPoints(rect)
            # boxes.append(np.int0(box))

            # Draw the contour and rotated rectangle
            cv2.drawContours(temp_image, contours, i, (0, 255, 0), 3)
            cv2.rectangle(temp_image, (x, y), (x + w, y + h), (0, 0, 255), 3)

            cropped_image = image[y:y + h, x:x + w]

            # DETECTION PART
            found = 0  # How many people is to be found in the contour
            confidence_sum = 0
            for detection in detect(model, cropped_image, label_map=label_map, min_score=args.confidence):
                score = detection['score'] * 100
                label = detection['label_name']
                annotation_label = '{} {:.2f}%'.format(label, score)
                annotate_image(cropped_image, detection['window'], label=annotation_label)
                if label == "person":
                    print("Found a person with confidence {}".format(score))
                    confidence_sum += score
                    found += 1

            # TODO: Check the contours on that area
            # Save the detection result
            # If a person is found in the cropped image, save the image
            if found > 0:
                import os

                if not os.path.exists("people"):
                    os.makedirs("people")

                file_name = "people\\f_{}-conf_{:.3f}.jpg".format(frameNum,
                            confidence_sum / found)
                cv2.imwrite(file_name, cropped_image)

            frameNum += 1

            # REDUNDANT
            # cv2.drawContours(image, boxes, i, (0, 255, 0), 3)
            # #Rotate the rectangle to normal rectangle for detection
            # rect  = rects[i]
            # angle = rect[-1]
            #
            # # http: // felix.abecassis.me / 2011 / 10 / opencv - rotation - deskewing /
            # if (angle < -45):
            #     angle += 90
            #     #Swap the width and the height
            #     size = list(map( lambda x: (int(x[1]),int(x[0])), [rect[1]]))[0]
            #
            # rotation_mat = cv2.getRotationMatrix2D(size, angle,1.0)
            # rotated_image = cv2.warpAffine(image,rotation_mat,image.shape[:2])
            # cropped = cv2.getRectSubPix(rotated_image, size, rect[0])
            #
            # cv2.imshow(WINDOW_NAME, cropped)
            # key = cv2.waitKey(0)

            # Detect the pedestrians


        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
        image[foreground_mask > 0] = [0,255,0]
        cv2.imshow(WINDOW_NAME,image)
        key = cv2.waitKey(5)


    cap.release()
