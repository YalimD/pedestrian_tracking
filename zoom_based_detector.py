#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:50:19 2018

@author: yalim
@coauthor: serkan
"""

#How deep we must go ?
#   expected#Detection   |    maxDepth      |    result
#         low            |      low         |   Won't go deep, finds large pedestrians
#         low            |      high        |   Might stop at high levels, saves time
#         high           |      low         |   Will search for all depth, optimal
#         high           |      high        |   Will take long, noisy results are possible

#TODO: Noisy detections needs to be removed.
#A detection is to be considered noisy if its size is inconsistent with the others and its ratio
#is not vertical (as other pedestrians) In case of multiple pedestrians, the detection doesn't really detect multiple
#people as a single pedestrian.
#In HOG, the pedestrians are assumed to be in ratio of 0.5
#Hold a gaussian for the detections so far, and label everything outside of 2std is an outlier.


import os
import cv2
import argparse
import numpy as np

from skimage import measure
from datetime import datetime
from detection_lib import *

class FrameGenerator:

    def __init__(self, videoName = 0):
        self.videoName = videoName
        if videoName.isdigit() and int(videoName) == 0:
            self.videoName = 0
            print("Playing from default source")
        else:
            print("Playing from source {}".format(videoName))
        self.cap = cv2.VideoCapture(self.videoName)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next().__next__()

    def next(self):

        while self.cap.isOpened():
            success, image = self.cap.read()  # Get frame from video capture

            # Bail out if we cannot read the given source
            if not success:
                print('Cannot read frame from source {}'.format(self.videoName))
                raise StopIteration
            yield image

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Process of {} is completed".format(self.videoName))
        self.cap.release()


class QuadDetectionTree:

    maxDepth = 2
    minImage = 4
    # Min number of detections in order to stop going deeper in the tree
    detectionThreshold = 1
    areaPercentage = 0.75

    # Put model as static
    model = None
    labelmap = None
    minConfidence = None

    def __init__(self, image, depth = 0):
        self.image = image
        self.depth = depth

    @classmethod
    def assignDetector(cls,model, labelmap, minConfidence):
        QuadDetectionTree.model = model
        QuadDetectionTree.labelmap = labelmap
        QuadDetectionTree.minConfidence = minConfidence

    @staticmethod
    def encapsulates(parent, child):

        childArea = (child[2] - child[0]) * (child[3] - child[1])
        #First, check if the parent and child intersects
        left = max(parent[0],child[0])
        right = min(parent[2],child[2])

        top = max(parent[1],child[1])
        bottom = min(parent[3],child[3])

        return left < right \
            and top < bottom \
            and (right - left) * (bottom - top) >= childArea * QuadDetectionTree.areaPercentage


    def processTile(self):

        # DETECTION PART
        confidence_sum = 0

        detections = []
        for detection in detect(QuadDetectionTree.model, self.image, label_map=QuadDetectionTree.labelmap,
                                min_score=QuadDetectionTree.minConfidence):
            score = detection['score'] * 100
            label = detection['label_name']

            if label == "person":
                #TODO: Why a considerable amount of noise appears on shadow areas
                annotation_label = '{} {:.2f}%'.format(detection['label_name'], detection['score'] * 100)
                annotate_image(self.image, detection['window'], label=annotation_label)
                # print("Found a person with confidence {} at {}".format(score, detection['window']))
                confidence_sum += score
                detections.append(detection['window']) #Need a rectangle

        # print("In frame with size {}x{}, detector found {} people with average confidence of {} "
        #       .format(self.frame.shape[0], self.frame.shape[1], len(detections), confidence_sum / max(len(detections),1)))

        # Create and run children if depth not reached to threshold and size of the frame is not too small
        if len(detections) < QuadDetectionTree.detectionThreshold \
                and self.depth < QuadDetectionTree.maxDepth \
                and self.image.shape[0] * self.image.shape[1] > QuadDetectionTree.minImage:

            row = self.image.shape[0]
            col = self.image.shape[1]

            top = slice(0,row//2)
            bottom = slice(row//2,row)
            left = slice(0,col//2)
            right = slice(col//2,col)

            tiles = [[top,left],
                     [top,right],
                     [bottom,left],
                     [bottom,right]]

            childrenDetections = []

            #Traverse children
            for tile in tiles:
                child = QuadDetectionTree(self.image[tile[0],tile[1]],depth=self.depth+1)
                child_result = child.processTile()
                if len(child_result) != 0:
                    #Adjust children's corners
                    for result in child_result:
                        childrenDetections.append((result[0] + tile[1].start,
                                                   result[1] + tile[0].start,
                                                   result[2] + tile[1].start,
                                                   result[3] + tile[0].start))

            extra_children = []

            #If no parent is found, directly add all children
            if len(detections) == 0:
                extra_children = childrenDetections

            else:
                #Erase children who are already included in this depth's detection
                for parent_det in detections:
                    for child in childrenDetections:
                        if not QuadDetectionTree.encapsulates(parent_det,child):
                            #If child is not inside of the parent
                            extra_children.append(child)
                        else:
                            print("A child at level {} is deleted".format(self.depth))
            detections = detections + extra_children

        return detections



#ZOOM based detector must search in windows in order to find the pedestrians, to find stationary ones too.
#Zoom hierarchy as a quad tree needs to be built
#When a ped is found on multiple tiles, we need to combine them using the center and similarity (histogram)
#of those pedestrians.
#Advantages to contour based:
#   Stationary ped can be found
#   May need contours to find the major axis, but only for found peds

#Pseudo algo
# for 1 to depthLimit
#   detect ped in current frame
#   If num of detected / size of frame < threshold (in contrary to classic quad)
#       keep the detections in this node and divide the tree
#   Combine detection windows from children (parts of pedestrians in lower levels)
#   (If child rect intersects with parent rect above certain % of its area, delete child)

if __name__ == "__main__":
    WINDOW_NAME = 'Detection Output'

    # Parse inputs
    parser = argparse.ArgumentParser(
        description="Runs model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-s', '--source', help="Source of the video", default='0')
    parser.add_argument('-c', '--confidence', help="Detection confidence", type=float, default=0.10)
    parser.add_argument('--maxdepth', help="Max depth of quad search tree", type=int, default=3)
    parser.add_argument('--minImage', help="Minimum frame size to be searched", type=int, default=1000)
    parser.add_argument('--areaPercentage',help="Percentage of the area of children to be considered as"
                                                "inside of the parent detection", type=float, default=0.25)
    parser.add_argument('--frameperdetection',help="Number of frames passed for each detection",
                        type=int, default=5)
    parser.add_argument('--minDetection', help="# of minimum detection to stop at a level",
                        type=int, default=1)

    args = parser.parse_args()

    # Load model
    detector = rnn_detection.RNN_Detector("mobilenet")

    # Assign it to quadtree and adjust its parameters
    QuadDetectionTree.assignDetector(detector.model, detector.label_map, args.confidence)
    QuadDetectionTree.maxDepth = args.maxdepth
    QuadDetectionTree.minImage = args.minImage
    QuadDetectionTree.detectionThreshold = args.minDetection
    QuadDetectionTree.areaPercentage = args.areaPercentage

    # Initialize video capture
    videoCap = FrameGenerator(args.source)

    prev_time = 0
    mean_fps = 0
    frameNum = 1
    detection_per_frame = args.frameperdetection
    for image in videoCap:

        # TODO: Correct the Time calculation
        cur_time = datetime.now().microsecond
        time_elapsed = cur_time - prev_time  # In seconds
        cur_fps = 1000000.0 / max(1, time_elapsed)
        mean_fps = mean_fps * 0.5 + cur_fps * 0.5
        prev_time = cur_time

        if frameNum % detection_per_frame == 0:

            #Generate the quadtree from the frame
            quad = QuadDetectionTree(image)
            detections = quad.processTile()

            for detection in detections:
                cv2.rectangle(image,(int(detection[0]),int(detection[1])),
                              (int(detection[2]),int(detection[3])), (0,0,255), thickness=3)

            cv2.putText(image, "FPS: {}".format(mean_fps), (10, 15),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (100, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, image)
            key = cv2.waitKey(2)

            #TODO: Turn this into a videowriter
            file_name = "zoomed_result\\f_{}.jpg".format(frameNum)
            cv2.imwrite(file_name, image)

        frameNum += 1




