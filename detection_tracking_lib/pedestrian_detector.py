import cv2
import numpy as np

from background_subtractor import *
from detection_tracking_lib import *

__all__ = ["PedestrianDetector"]

class PedestrianDetector:


    def __init__(self, detector_folder, confidence=0.15, hogParameters = {},
                 backgroundsubtraction='mog', stabilizer='lk',
                 morph_kernel_size=5, contour_threshold=100, box_margin=100, det_out_name = None):

        # Create the model object by giving the name of the folder where its model exists
        if detector_folder.lower() != "hog":
            print("Using RNN ({})".format(detector_folder))
            self.detector = rnn_detection.RNN_Detector(detector_folder)
        else:
            print("Using HOG")
            self.detector = HogDetector(hogParameters)

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

        #Create a txt file for detection results
        if det_out_name is not None:
            self.det_output = open(det_out_name,mode="w")
        else:
            self.det_output = None

    def closeFile(self):
        if self.det_output is not None:
            self.det_output.close()

    def processImage(self, frame, frameID, prev_mask, removeShadows = False):

        if removeShadows:
            from shadow_remover import ShadowRemoval
            #Shadow Removal causes HOG to have higher FP but improves RNN
            frame = ShadowRemoval.ShadowRemover.removeShadows(frame)

        # Apply the read frame to the background subtractor and obtain foreground mask and the stabilized frame
        foreground_mask, stabilized_frame = self.backgroundSubtractor.apply(frame)

        # Now search for pedestrians in each contour in the background subtracted frame
        # Each contours's bounding box is searched

        # Apply dilation, erosion
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
        opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
        cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, closing_kernel, foreground_mask)
        cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, opening_kernel, foreground_mask)


        #Draw the foreground + previous mask (previous mask is used to search areas where trackers are, even if
        # there is currently no movement ) as green
        prev_mask[foreground_mask > 0] = 1
        # stabilized_frame[foreground_mask > 0, 1] = 255

        temp_image, contours, hierarchy = cv2.findContours(prev_mask, cv2.RETR_EXTERNAL,
                                                           cv2.CHAIN_APPROX_TC89_KCOS)

        contours = list(filter(lambda c: cv2.contourArea(c) > self.contour_threshold, contours))

        detections = []
        scores = []  # Unused for Hog
        # Define bounding boxes of the contours
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            box_margin = self.box_margin
            x = max(x - box_margin, 0)
            y = max(y - box_margin, 0)
            w = min(w + box_margin * 2, stabilized_frame.shape[1])
            h = min(h + box_margin * 2, stabilized_frame.shape[0])

            cropped_image = stabilized_frame[y:y + h, x:x + w]

            # # When we apply the HOG over all of the image, process slows down significantly
            # cropped_image = np.copy(stabilized_frame)
            # x = 0; y = 0; w = stabilized_frame.shape[1]; h = stabilized_frame.shape[0]

            # RNN DETECTION
            if type(self.detector) == rnn_detection.RNN_Detector:


                for detection in self.detector.detect(cropped_image, min_score=self.confidence):
                    score = detection['score'] * 100
                    label = detection['label_name']

                    if label == "person":
                        #Adjust the coordinate of the window according to whole image
                        det_win = detection['window']
                        det_win = (int(det_win[0] + x), int(det_win[1] + y), int(det_win[2] + x), int(det_win[3] + y))

                        detections.append(det_win)
                        scores.append(score)

            # HOG DETECTION
            elif type(self.detector) == HogDetector:

                #Resize the image few times in order get a better detection
                resize_cropped = 3 # Parameter
                cropped_image = cv2.resize(cropped_image,None,None,fx=resize_cropped,fy=resize_cropped)

                # Parameter
                raw_detections, score = self.detector.detector.detectMultiScale(cropped_image,
                                                                                  hitThreshold=0,
                                                                                  winStride=(8,8),
                                                                                  padding=(8,8),
                                                                                  scale=1.2,
                                                                                  finalThreshold=1.5,
                                                                                  useMeanshiftGrouping=False)
                if len(raw_detections) > 0:
                    # print("HOG Detections {}".format(raw_detections))

                    for detection in raw_detections:

                        # Parameter
                        hog_adjustment_scale = 0.1

                        detection[0] += (hog_adjustment_scale / 2) * detection[2]
                        detection[1] += (hog_adjustment_scale / 2) * detection[3]
                        detection[2] = (1 - hog_adjustment_scale) * detection[2]
                        detection[3] = (1 - hog_adjustment_scale) * detection[3]


                        det_win = (int((detection[0] / resize_cropped) + x), int((detection[1] / resize_cropped) + y),
                                   int(((detection[0] + detection[2]) / resize_cropped) + x),
                                   int(((detection[1] + detection[3]) / resize_cropped) + y))

                        detections.append(det_win)
                    scores = np.append(scores,score)
        # Combine detections
        detections = PedestrianDetector.combineDetectionWindowsNMS(detections,scores)
        # PedestrianDetector.combineDetectionWindowsGreedly(detections)

        #Draw detection boxes
        for detection in detections:
            cv2.rectangle(stabilized_frame,tuple(map(int,detection[0:2])),tuple(map(int,detection[2:])),(0,255,255),2)

        if self.det_output is not None:
            for detection in detections:

                #Some preprocessing before writing to txt
                bb_left = detection[0]
                bb_top = detection[1]
                bb_width = detection[2] - detection[0]
                bb_height =  detection[3] - detection[1]

                self.det_output.write("{},{},{},{},{},{},{}\n".format(frameID, -1, bb_left, bb_top,
                                                                          bb_width, bb_height, 1))

        #Foreground mask only contains the postures of the detected pedestrians
        return stabilized_frame, detections, foreground_mask

    #Combine detection windows if they intersect over a certain area percentage
    @staticmethod
    def combineDetectionWindowsGreedly(detections):
        det_index = 0
        # print("Before combining boxes {}".format(len(detections)))
        while det_index < len(detections)-1:
            test_index = det_index+1

            while test_index < len(detections):
                # Combine the detections, None if they don't encapsulate each other
                combination = PedestrianDetector.combine_rect(detections[det_index], detections[test_index])
                if combination:
                    del detections[test_index]; del detections[det_index]
                    detections.insert(det_index,combination)
                    det_index -= 1
                    break
                test_index += 1
            det_index += 1

    # https://www.vision.ee.ethz.ch/publications/papers/proceedings/eth_biwi_01126.pdf
    #Filter detection windows according to their scores and area intersection ?
    @staticmethod
    def combineDetectionWindowsNMS(detections,scores):

        #Create a temporary list of detections adjusted by their width-height rather than lower right corner
        temp_det = []
        for detection in detections:
            temp_det.append((detection[0],detection[1],detection[2] - detection[0], detection[3] - detection[1]))

        # I think the nms_threshold here is the area threshold
        indices = cv2.dnn.NMSBoxes(temp_det,scores,score_threshold=0,nms_threshold=0.6)

        return [detections[i] for i in list(map(int,list(indices)))]

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
            and (right - left) * (bottom - top) >= min(r1_area,r2_area) * 1: # Parameter

            return (min(large[0], small[0]), min(large[1], small[1]), max(large[2], small[2]), max(large[3], small[3]))

        return None

class HogDetector:

    # Parameter
    def __init__(self, parameters = {}):
        self.detector = cv2.HOGDescriptor()
        self.detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
