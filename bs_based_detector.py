import os
import cv2
import argparse
import numpy as np
import imutils

from skimage import measure
from datetime import datetime
from background_subtractor import *
from detection_lib import *
from imutils.video import VideoStream

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

#TODO: Include multiple types of detectors
class PedestrianDetector:


    def __init__(self, detector_folder, confidence=0.15,
                 backgroundsubtraction='mog', stabilizer='lk',
                 morph_kernel_size=5, contour_threshold=100, box_margin=100):

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

            # Draw the rectangles TODO: Remove
            # cv2.rectangle(stabilized_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

            cropped_image = frame[y:y + h, x:x + w]

            # DETECTION PART
            for detection in self.detector.detect(cropped_image, min_score=self.confidence):
                score = detection['score'] * 100
                label = detection['label_name']

                if label == "person":

                    #Adjust the coordinate of the window according to whole image
                    det_win = detection['window']
                    det_win = ((int(det_win[0] + x), int(det_win[1] + y),int(det_win[2] + x), int(det_win[3] + y)), score)

                    detections.append(det_win)

        #Combine detections
        PedestrianDetector.combineDetectionWindows(detections)

        for detection in detections:
            annotation_label = '{:.2f}%'.format(detection[1])
            self.detector.annotate_image(stabilized_frame, detection[0],
                                         label=annotation_label)
            # print("Found a person with confidence {}".format(detection[1]))

        return stabilized_frame

    #Combine detection windows if they intersect over a certain area percentage
    @staticmethod
    def combineDetectionWindows(detections):
        det_index = 0
        # print("Before combining boxes {}".format(len(detections)))
        while det_index < len(detections)-1:
            test_index = det_index+1

            while test_index < len(detections):
                # Combine the detections, None if they don't encapsulate each other
                combination = PedestrianDetector.combine_rect(detections[det_index][0], detections[test_index][0])
                if combination:
                    combination = (combination, detections[det_index][1])
                    del detections[test_index]; del detections[det_index]
                    detections.insert(det_index,combination)
                    det_index -= 1
                    break
                test_index += 1
            det_index += 1

        # print("After combining boxes {}".format(len(detections)))


    @staticmethod
    def rect_area(rect):
        return (rect[2] - rect[0]) * (rect[3] - rect[1])

    @staticmethod
    #Returns the combination of rectangles if one encapsulates the other for certain percentage
    def combine_rect(rect1, rect2):

        r1_area = PedestrianDetector.rect_area(rect1)
        r2_area = PedestrianDetector.rect_area(rect2)

        large,small = (rect1,rect2) if r1_area >= r2_area else (rect2,rect1)

        #First, check if the parent and child intersects
        left = max(large[0],small[0])
        right = min(large[2],small[2])

        top = max(large[1],small[1])
        bottom = min(large[3],small[3])

        if left < right \
            and top < bottom \
            and (right - left) * (bottom - top) >= min(r1_area,r2_area) * 0.3: #TODO: Parameter

            return (min(large[0], small[0]), min(large[1], small[1]), max(large[2], small[2]), max(large[3], small[3]))

        return None

if __name__ == "__main__":
    WINDOW_NAME = 'Detection Output'

    # Parse inputs
    parser = argparse.ArgumentParser(
        description="Runs model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-s', '--source', help="Source of the video", default='0')
    parser.add_argument('-o', '--output', help='Name of the output video (with detection)', default='output.mp4')
    parser.add_argument('-d', '--detector', help="The detector to be used (if rnn, pass the folder containing the related"
                                                 "graph files)", default='mobilenet')
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
    # TODO: Side by side comparison of detections
    with VideoReader(source) as cap:

        ped_detector = PedestrianDetector(args.detector)

        initiation = datetime.now().timestamp()
        num_of_frames = 0
        prev_time = initiation
        mean_fps = 0

        out_fps = 30 if (source == 0) else cap.get(cv2.CAP_PROP_FPS)

        writer = cv2.VideoWriter(args.output, int(cap.get(cv2.CAP_PROP_FOURCC)),
                                 out_fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        if writer.isOpened():
            print("Started writing the output to {}".format(args.output))
        else:
            print("Failed to start writing output to {}".format(args.output))
            raise IOError

        while True:

            # Time calculation
            cur_time = datetime.now().timestamp()
            time_elapsed = cur_time - prev_time  # In seconds
            cur_fps = 1 / time_elapsed
            mean_fps = mean_fps * 0.5 + cur_fps * 0.5
            prev_time = cur_time

            print("Mean FPS: {}".format(mean_fps))
            success, frame = cap.read()  # Get frame from video capture
            num_of_frames += 1

            # Bail out if we cannot read the frame
            if success == False:
                print('Cannot read frame from source {}'.format(source))
                break

            frame = ped_detector.processImage(frame)

            cv2.putText(frame, "FPS: {:.2f}".format(mean_fps), (10, 15),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (100, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(5)

            if key == 27 or key == ord('q'):
                break

            writer.write(frame)

    if writer:
        writer.release()

    print("It took {} seconds for the program to process the give video with {} number of frames".format(
        datetime.now().timestamp() - initiation, num_of_frames))
