import os
import cv2
import argparse
import numpy as np

from skimage import measure
from datetime import datetime
from pedestrian_tracker import *
from detection_lib import *

"""
Created on Tue Apr 24 17:50:19 2018

@author: yalim
@coauthor: serkan
"""

#NOTES:
#As it appears, when the pedestrian is occluded (by a tree) the detection window includes the
#large portion of the tree. In our case, it even includes the pedestrian's shadow
#30 confidence includes shadows and some uncomprehensible images
#33 seems like a sweet spot

#TODO:Find the major axes of detected pedestrians using thresholding etc. (back sub ?)
#If above fails (or doesn't work well, just use the lines in the frame to find zenith)
#TODO: Try with different models (this must be done while we are writing)
#TODO: Video Output

class VideoReader:
    def __init__(self, source):
        self.source = source

    def __enter__(self):
        self.video_capture = cv2.VideoCapture(self.source)

        if not self.video_capture.isOpened():
            raise IOError('Cannot open video file')

        return self.video_capture

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.video_capture.release()

class PedestrianTracker:


    def __init__(self, detector_folder, confidence=0.25,
                 backgroundsubtraction='mog', stabilizer='lk',
                 morph_kernel_size=5, contour_threshold=100, box_margin=20):

        # Create the model object by giving the name of the folder where its model exists
        self.detector = rnn_detection.RNN_Detector(detector_folder)
        self.confidence = confidence

        # Initialize the background subtractor
        # CNT is too noisy but can be cleared using erosion
        # GSOC has a lot of initial noise, but gets better in time
        # LSBP has a lot of noise
        # MOG is improved compared to the previous version
        self.backgroundSubtractor = bgsegm.BackgroundSubtractor(method=backgroundsubtraction,
                                                                stabilizer=stabilizer)

        self.morph_kernel_size = morph_kernel_size
        self.contour_threshold = contour_threshold
        self.box_margin = box_margin


    def processImage(self, image):

        # Apply the read frame to the background subtractor and obtain foreground mask and the stabilized frame
        foreground_mask, stabilized_frame = self.backgroundSubtractor.apply(frame)

        # Now search for pedestrians in each contour in the background subtracted frame
        # Each contours's bounding box is searched

        # Apply dilation, erosion
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
        opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
        cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, closing_kernel, foreground_mask)
        cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, opening_kernel, foreground_mask)

        #Draw the foreground as blue
        stabilized_frame[foreground_mask > 0, 0] = 255

        temp_image, contours, hierarchy = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL,
                                                           cv2.CHAIN_APPROX_TC89_KCOS)

        contours = list(filter(lambda c: cv2.contourArea(c) > self.contour_threshold, contours))

        detections = []
        # Define bounding boxes of the contours
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])

            box_margin = self.box_margin
            x = max(x - box_margin, 0)
            y = max(y - box_margin, 0)
            w = min(w + box_margin * 2, stabilized_frame.shape[1])
            h = min(h + box_margin * 2, stabilized_frame.shape[0])

            # Draw the rectangles

            cv2.rectangle(stabilized_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

            cropped_image = frame[y:y + h, x:x + w]

            # DETECTION PART

            for detection in self.detector.detect(cropped_image, min_score=self.confidence):
                score = detection['score'] * 100
                label = detection['label_name']

                if label == "person":
                    detections.append((detection['window'],score))

        # filtered_detections = []
        # for detection in detections:
        #     child = False
        #     for test_detection in detections:
        #         if test_detection is not detection:
        #             encapsulated = PedestrianTracker.encapsulates(detection[0], test_detection[0])
        #             if encapsulated is detection[0]:
        #                 child = True
        #                 break
        #     if not child:
        #         filtered_detections.append(detection)

        for detection in detections:
            annotation_label = 'Ped {:.2f}%'.format(score)
            self.detector.annotate_image(stabilized_frame, detection[0],
                                         label=annotation_label,
                                         window_corner=(x, y))
            print("Found a person with confidence {}".format(detection[1]))

        return stabilized_frame

    @staticmethod
    def rect_area(rect):
        return (rect[2] - rect[0]) * (rect[3] - rect[1])

    @staticmethod
    def encapsulates(rect1, rect2):

        r1_area = PedestrianTracker.rect_area(rect1)
        r2_area = PedestrianTracker.rect_area(rect2)

        large,small = (rect1,rect2) if r1_area >= r2_area else (rect2,rect1)

        #First, check if the parent and child intersects
        left = max(large[0],small[0])
        right = min(large[2],small[2])

        top = max(large[1],small[1])
        bottom = min(large[3],small[3])

        if left < right \
            and top < bottom \
            and (right - left) * (bottom - top) >= min(r1_area,r2_area) * 0.4:

            return small

        return None


if __name__ == "__main__":
    WINDOW_NAME = 'Detection Output'

    # Parse inputs
    parser = argparse.ArgumentParser(
        description="Runs model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-s', '--source', help="Source of the video", default='0')
    parser.add_argument('-c', '--confidence', help="Detection confidence", type=float, default=0.15)
    #TODO: Background subtraction and stabilization parameters needs to be added
    args = parser.parse_args()

    if (args.source and args.source != '0'):
        print("Playing from source {}".format(args.source))
        source = args.source
        if (source.isdigit()):
            source = int(source)
    else:
        source = 0
        print("Playing from default source")


    # Initialize and run video capture
    with VideoReader(source) as cap:

        ped_tracker = PedestrianTracker('mobilenet')

        prev_time = 0
        mean_fps = 0
        frameNum = 0

        while True:

            # Time calculation
            cur_time = datetime.now().microsecond
            time_elapsed = cur_time - prev_time  # In seconds
            cur_fps = 1000000.0 / time_elapsed
            mean_fps = mean_fps * 0.5 + cur_fps * 0.5
            prev_time = cur_time

            success, frame = cap.read()  # Get frame from video capture

            # Bail out if we cannot read the frame
            if success == False:
                print('Cannot read frame from source {}'.format(source))
                break

            frame = ped_tracker.processImage(frame)

            # Save the detection result
            # If a person is found in the cropped frame, save the frame
            # if found > 0:
            #     import os
            #
            #     if not os.path.exists("people"):
            #         os.makedirs("people")
            #
            #     file_name = "people\\f_{}-conf_{:.3f}.jpg".format(frameNum,
            #                 confidence_sum / found)
            #     cv2.imwrite(file_name, cropped_image)

            frameNum += 1

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(5)
