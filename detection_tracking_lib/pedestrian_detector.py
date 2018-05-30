import cv2

from background_subtractor import *
from detection_tracking_lib import *

__all__ = ["PedestrianDetector"]

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


    def processImage(self, frame):

        #TODO: Apply shadow remover ?

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

            cropped_image = frame[y:y + h, x:x + w]

            # DETECTION PART
            for detection in self.detector.detect(cropped_image, min_score=self.confidence):
                score = detection['score'] * 100
                label = detection['label_name']

                if label == "person":

                    #Adjust the coordinate of the window according to whole image
                    det_win = detection['window']
                    #TODO: Score might be removed
                    # det_win = ((int(det_win[0] + x), int(det_win[1] + y),int(det_win[2] + x), int(det_win[3] + y)), score)
                    det_win = (int(det_win[0] + x), int(det_win[1] + y), int(det_win[2] + x), int(det_win[3] + y))

                    detections.append(det_win)

        #Combine detections
        PedestrianDetector.combineDetectionWindows(detections)

        # TODO: Need to delete this one now, leaving for debugging
        for detection in detections:
            annotation_label = '{:.2f}%'.format(detection[1])

            self.detector.annotate_image(stabilized_frame, detection,
                                         label=annotation_label)
            # print("Found a person with confidence {}".format(detection[1]))

        return stabilized_frame, detections

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