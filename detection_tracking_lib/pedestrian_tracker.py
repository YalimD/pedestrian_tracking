from enum import IntEnum
from skimage import measure

import cv2
import numpy as np


__all__ = ["MultiPedestrianTracker"]

class TrackState(IntEnum):
    START = 1
    INACTIVE = 2
    ACTIVE = 3
    DEAD = 4

class PedestrianTracker:
    TRACKER_MAX_DISTANCE = 50
    TRACKER_ACTIVATE_COUNT = 5
    TRACKER_DEATH_COUNT = 300
    TRACKER_INACTIVATE_COUNT = 10
    TRACKER_SCORE_MIX_RATIO = 0.8 #0.5


    def __init__(self, initial_bounding_box, id,
                 feature=None):

        self.detected = True
        self.id = id

        self.time_passed_since_correction = 0
        self.correction_count = 0  # Number of times the pedestrian has been detected
        self.state = TrackState.START

        self.scale_x = initial_bounding_box[2] - initial_bounding_box[0]
        self.scale_y = initial_bounding_box[3] - initial_bounding_box[1]

        # Visual identity of the tracked pedestrian
        if feature is not None:
            self.color_data = np.copy(feature)

        # The associated Kalman Filter
        self.filter = None
        self.__initializeKalmanFilter(initial_bounding_box) # Next position of the pedestrian is predicted

        if feature is not None:
            self.correct(initial_bounding_box, [(),()], feature)

        self.head_feet_pos = [(),()] #Unused in calculations, but written to the output file

    # Using Kalman Filter, predicts the next position, updates the tracker's state accordingly
    def predict(self):
        self.detected = False

        self.time_passed_since_correction += 1

        if self.time_passed_since_correction >= PedestrianTracker.TRACKER_INACTIVATE_COUNT \
                and self.state == TrackState.ACTIVE:
            self.state = TrackState.INACTIVE
        if self.time_passed_since_correction >= PedestrianTracker.TRACKER_DEATH_COUNT // 50 \
                and self.state == TrackState.START:
            self.state = TrackState.DEAD
        if self.time_passed_since_correction >= PedestrianTracker.TRACKER_DEATH_COUNT:
            self.state = TrackState.DEAD

        prediction = self.filter.predict()

        return prediction

    def correct(self, bounding_box, head_feet_pos, feature = None):

        self.detected = True
        self.correction_count += 1
        self.time_passed_since_correction = 0

        self.scale_x = (bounding_box[2] - bounding_box[0]) * 0.5 + self.scale_x * 0.5
        self.scale_y = (bounding_box[3] - bounding_box[1]) * 0.5 + self.scale_y * 0.5

        position = PedestrianTracker.getCenter(bounding_box)

        # (Re)activate the tracker if correction count exceeds a certain threshold
        if self.correction_count >= PedestrianTracker.TRACKER_ACTIVATE_COUNT \
                and (self.state == TrackState.INACTIVE or self.state == TrackState.START):
            self.state = TrackState.ACTIVE

            self.__initializeKalmanFilter(bounding_box)
            self.predict()

        measurement = np.array(position,dtype=np.float32)

        # Update the predicted state of the kalman filter using the measurements
        self.filter.correct(measurement)

        if feature is not None:
            # If the tracker is active, update the visual identity
            if self.state != TrackState.INACTIVE:
                blend_alpha = 0.4  # Parameter
                cv2.addWeighted(self.color_data, blend_alpha,
                                feature, 1 - blend_alpha,0, self.color_data)

        if head_feet_pos is not None:
            self.head_feet_pos = head_feet_pos
        else: #If the found orientation is too noisy, consider bounding box version of head/feet; top and bottom center
            center = self.getRect()
            self.head_feet_pos = [(center[0], center[1] - self.scale_y / 2),(center[0], center[1] + self.scale_y / 2)]

    # Returns cost
    # Cost should be between 0 and 1
    # 0 -> means detected object probably is the tracked object
    # 1 -> means it's not
    def cost(self, bounding_box, feature = None):
        position = self.getCenter(bounding_box)
        prediction = [self.filter.statePre[0], self.filter.statePre[1]]

        positional_cost = np.linalg.norm(np.array(prediction) - np.array(position))
        positional_cost = 1 if positional_cost > PedestrianTracker.TRACKER_MAX_DISTANCE \
            else positional_cost / PedestrianTracker.TRACKER_MAX_DISTANCE

        # Distance between features in sense of pixels
        if feature is not None:
            similarity_cost = cv2.norm(feature, self.color_data, cv2.NORM_L2)
            return (positional_cost * PedestrianTracker.TRACKER_SCORE_MIX_RATIO) + \
                   (similarity_cost * (1.0 - PedestrianTracker.TRACKER_SCORE_MIX_RATIO))
        return positional_cost

    # Returns the estimated velocity of the target
    def getVelocity(self):
        return (self.filter.statePost[2], self.filter.statePost[3])

    # Returns the estimated bounding box
    def getRect(self):
        last_position = (self.filter.statePost[0], self.filter.statePost[1])
        return (last_position[0] - self.scale_x / 2, last_position[1] - self.scale_y / 2,
                last_position[0] + self.scale_x / 2, last_position[1] + self.scale_y / 2)

    def getHeadFeet(self):
        return self.head_feet_pos

    # Returns the center of the bounding box
    @staticmethod
    def getCenter(rect):
        return (rect[0] + (rect[2] - rect[0])/2, (rect[1] + (rect[3] - rect[1])/2))

    def __initializeKalmanFilter(self, initial_bounding_box):
        self.filter = cv2.KalmanFilter(4, 2, 0)
        self.filter.transitionMatrix = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0],
                                        [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], np.float32)

        initial_position = self.getCenter(initial_bounding_box)

        self.filter.statePre = np.array([initial_position[0], initial_position[1], 0, 0],np.float32)
        self.filter.statePost = np.array([initial_position[0], initial_position[1], 0, 0],np.float32)

        self.filter.measurementMatrix = np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0]],np.float32)
        # cv2.setIdentity(self.filter.measurementMatrix)
        cv2.setIdentity(self.filter.processNoiseCov, 10 ** -4)
        cv2.setIdentity(self.filter.measurementNoiseCov, 10 ** -1)
        cv2.setIdentity(self.filter.errorCovPost, 10 ** -1)




class MultiPedestrianTracker:
    MULTI_TRACKER_ASSOCIATION_THRESHOLD = 0.4
    HISTOGRAM_SIZE = 16
    USE_FEATURES = True

    def __init__(self, detector, removeShadows, bounding_box_file = None, head_feet_file = None):

        self.id_generator = MultiPedestrianTracker.next_id()
        self.detector = detector
        self.trackers = []
        self.removeShadows = removeShadows

        self.bb_file = bounding_box_file
        self.hf_file = head_feet_file


    @staticmethod
    def next_id():
        ped_id = 0
        while True:
            ped_id += 1
            yield ped_id

    #Extracts the features inside the detection windows as histograms
    @staticmethod
    def extractFeatures(stabilized_frame, detections):

        features = []
        #For each detection window, extract features
        for detection in detections:

            t_detection = np.array(detection)
            #Sometimes using inception, the detection window contains negative values
            t_detection[t_detection < 0] = 0

            area_OI = stabilized_frame[slice(t_detection[1],t_detection[3]),slice(t_detection[0],t_detection[2])]

            #Convert the area into LAB colorspace
            area_OI = cv2.cvtColor(area_OI,cv2.COLOR_BGR2LAB)

            num_of_channels = list(range(area_OI.shape[-1]))
            hist = []
            for c in num_of_channels:
                t = cv2.calcHist([area_OI], [c], None,
                                    [MultiPedestrianTracker.HISTOGRAM_SIZE],
                                    [0, 255])
                cv2.normalize(t, t, 1, 0, cv2.NORM_L2)
                hist.append(t)

            hist = cv2.vconcat(hist)
            hist = cv2.transpose(hist)
            # cv2.normalize(hist,hist,1,0,cv2.NORM_L2)
            features.append(hist)
        return features


    def predict(self):
        for tracker in self.trackers:
            tracker.predict()

    def update(self, frame, frameID):

        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype= np.uint8)
        for tracker in self.trackers:
            rect = tracker.getRect()
            mask[int(rect[1]):int(rect[3]),int(rect[0]):int(rect[2])] = 255

        #Get detection ( List of detections as boxes ) Also get the stabilized frame and mask associated with bs
        stabilized_frame, detections, f_mask = self.detector.processImage(frame, frameID, mask, self.removeShadows)

        #Extract the features for all detections. Which is in this case, is the area inside of their detection boxes
        features = None
        if MultiPedestrianTracker.USE_FEATURES:
            features = MultiPedestrianTracker.extractFeatures(stabilized_frame, detections)

        # Used to track which  tracker/ detection
        tracker_assigned = [False for i in self.trackers]
        detection_assigned = [False for i in detections]

        num_tracker_assigned = 0
        num_detection_assigned = 0

        num_trackers = len(self.trackers)
        num_detections = len(detections)

        # Step 1: Greedy association
        if num_detections != 0 and num_trackers != 0:

            # For each step find best tracker/detection pair and update tracker using detection
            while num_tracker_assigned < num_trackers and num_detection_assigned < num_detections:
                best_tracker = None
                best_detection = None

                best_tracker_index = -1
                best_detection_index = -1

                min_cost = np.inf

                # For each detection/tracker pair find best pairh e
                for i in range(num_detections):
                    if not detection_assigned[i]:
                        detection = detections[i]
                        for j in range(num_trackers):
                            if not tracker_assigned[j]:
                                tracker = self.trackers[j]
                                if features is not None:
                                    cost = tracker.cost(detection, features[i])
                                else:
                                    cost = tracker.cost(detection)
                                if cost < min_cost:
                                    min_cost = cost
                                    best_tracker = tracker
                                    best_detection = detection

                                    best_detection_index = i
                                    best_tracker_index = j

                # Update tracker using detection
                if best_detection_index == -1 or best_tracker_index == -1:
                    break

                # If quality of the best detection/tracker pair is not good enough
                #  Then stop association!!!
                if min_cost > MultiPedestrianTracker.MULTI_TRACKER_ASSOCIATION_THRESHOLD:
                    break


                # Use the associated detection to find the head and feet position of the pedestrian
                # This step is done before unassigned detections are associated with trackers, in order to care only for
                # "consistent" trackers
                # TODO: For both detections, we may need to increase the area we are searching for. By introducing a margin
                # Put the found values into the tracker, which will be added to the output file
                head_feet_pos = MultiPedestrianTracker.findHeadFeetPositions(frame, f_mask, best_detection)

                # Correct tracker using detection
                if features is not None:
                    best_tracker.correct(best_detection, head_feet_pos, features[best_detection_index])
                else:
                    best_tracker.correct(best_detection, head_feet_pos)

                # Update assignment stats
                detection_assigned[best_detection_index] = True
                tracker_assigned[best_tracker_index] = True
                num_tracker_assigned += 1
                num_detection_assigned += 1

        # Step 2: Create new trackers for unassigned detections
        #         Maybe create detection only if detection has high confidence

        if num_detections > 0 and num_detection_assigned < num_detections:
            for i in range(num_detections):
                if not detection_assigned[i]:
                    detection = detections[i]
                    if features is not None:
                        self.trackers.append(PedestrianTracker(detection, next(self.id_generator), features[i]))
                    else:
                        self.trackers.append(PedestrianTracker(detection, next(self.id_generator)))

        # Step 3: Handle unassigned trackers (Remove Dead ones)
        # Remove dead trackers
        self.trackers = list(filter(lambda t: t.state != TrackState.DEAD, self.trackers))

        #Return the stabilized frame together with the tracked bounding boxes
        return stabilized_frame

    # Finds the major axes as head and feet pos for each detection that is just associated to any tracker
    def findHeadFeetPositions(frame, f_mask, best_detection):

        detection_area = frame[best_detection[1]:best_detection[3],best_detection[0]:best_detection[2]]
        foreground_area = f_mask[best_detection[1]:best_detection[3],best_detection[0]:best_detection[2]]

        lab_detection = cv2.cvtColor(detection_area,cv2.COLOR_BGR2LAB)

        #Determine the threshold values from the bs result
        #Threshold values are determined as mean +- variance of values in each channel
        threshold_mean = [np.mean(lab_detection[foreground_area > 0, i]) for i in range(3)]
        threshold_variance = [(np.var(lab_detection[foreground_area > 0, i])) for i in range(3)]

        upper_threshold = tuple([sum(x) for x in zip(threshold_mean, 0*np.sqrt(threshold_variance))])
        lower_threshold = tuple([x[0] - x[1] for x in zip(threshold_mean, 2* np.sqrt(threshold_variance))])

        print("{} and {} thresholds".format(upper_threshold,lower_threshold))

        mask = cv2.inRange(lab_detection, (lower_threshold[0],0,0), (upper_threshold[0],255,255)) | foreground_area

        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel, mask)
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel, mask)

        #Determine the major axis of the largest blob and rotate the image to acquireit as a perpendicular line
        labels = measure.label(mask)
        try:
            biggest_label = np.bincount(labels.flatten())[1:].argmax() + 1
        except: #No regions found
            return [(), ()]

        labels[labels != biggest_label] = 0 #Only the biggest region
        region = measure.regionprops(labels)[0]

        #Find the head and feet positions approximately
        y0, x0 = region.centroid
        orientation = region.orientation
        head_x = x0 + np.cos(orientation) * 0.5 * region.major_axis_length
        head_y = y0 - np.sin(orientation) * 0.5 * region.major_axis_length
        feet_x = x0 - np.cos(orientation) * 0.5 * region.major_axis_length
        feet_y = y0 + np.sin(orientation) * 0.5 * region.major_axis_length

        mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

        cv2.circle(mask, (int(head_x), int(head_y)), 5, (0, 0, 255), -1)
        cv2.circle(mask, (int(feet_x), int(feet_y)), 5, (0, 0, 255), -1)

        #TODO: If the distance between points is below a threshold and orientation
        #is too off, then assume previous head-feet from tracker if any. Else assign top of
        #tracker box
        if False:
            return None


        cv2.imshow("The det area in lab", lab_detection)
        cv2.imshow("Original detection", detection_area)
        cv2.imshow("The mask", mask)
        cv2.imshow("Foreground pixels", foreground_area)
        cv2.waitKey(0)

        return [(head_x,head_y),(feet_x,feet_y)]

    def draw_and_write_trackers(self, frame, frameID, output_file = None):

        for tracker in self.trackers:
            if tracker.state == TrackState.ACTIVE:
                bounding_box = tracker.getRect()
                center = tuple(map(int,PedestrianTracker.getCenter(bounding_box)))
                velocity = list(map(int,tracker.getVelocity()))
                head_feet_pos = tracker.getHeadFeet()

                if frame is not None:
                    cv2.rectangle(frame,tuple(map(int,bounding_box[0:2])),tuple(map(int,bounding_box[2:])),
                                  (255,0,0), thickness=1)
                    cv2.arrowedLine(frame,center, tuple([c + v for c,v in zip(center, velocity)]),(0,0,255),3,tipLength=2)

                if output_file is not None:
                    # In append mode, write to the file if its given

                    #OLD VERSION
                    # output_file.write("{}, {}, {}, {}, {}, {}, {},".format(tracker.id,
                    #                                                        center[0],
                    #                                                        center[1],
                    #                                                        tracker.getVelocity()[0],
                    #                                                        tracker.getVelocity()[1],
                    #                                                        tracker.scale_x,
                    #                                                        tracker.scale_y))

                    #NEW, EVALUATION FRIENDLY VERSION
                    output_file.write("{},{},{},{},{},{},{},{},{}/{},{}/{} \n".format(frameID,
                                                                           tracker.id,
                                                                           bounding_box[0],
                                                                           bounding_box[1],
                                                                           bounding_box[2] - bounding_box[0],
                                                                           bounding_box[3] - bounding_box[1],
                                                                           -1,-1,
                                                                           head_feet_pos[0][0],head_feet_pos[0][1],
                                                                           head_feet_pos[1][0],head_feet_pos[1][1],
                                                                        ))




