from enum import Enum

import cv2
import numpy as np


__all__ = ["MultiPedestrianTracker"]

class TrackState(Enum):
    START = 1
    INACTIVE = 2
    ACTIVE = 3
    DEAD = 4

class PedestrianTracker:
    TRACKER_MAX_DISTANCE = 50
    TRACKER_ACTIVATE_COUNT = 15
    TRACKER_DEATH_COUNT = 300
    TRACKER_INACTIVATE_COUNT = 30
    TRACKER_SCORE_MIX_RATIO = 0.7

    def __init__(self, initial_bounding_box,
                 feature=None):

        self.detected = True
        self.id = next(PedestrianTracker.next_id())

        self.time_passed_since_correction = 0
        self.correction_count = 0  # Number of times the pedestrian has been detected
        self.state = TrackState.START

        self.scale_x = initial_bounding_box[2] - initial_bounding_box[0]
        self.scale_y = initial_bounding_box[3] - initial_bounding_box[1]

        # Visual identity of the tracked pedestrian
        if feature:
            self.color_data = np.copy(feature)

        # The associated Kalman Filter
        self.filter = None
        self.__initializeKalmanFilter(initial_bounding_box)
        self.predict()  # Next position of the pedestrian is predicted

        if feature:
            self.correct(initial_bounding_box, feature)

    @staticmethod
    def next_id():
        id = 0
        while True:
            id += 1
            yield id

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

    def correct(self, bounding_box, feature):

        self.detected = True
        self.correction_count += 1
        self.time_passed_since_correction = 0

        self.scale_x = bounding_box.width * 0.5 + self.scale_x * 0.5
        self.scale_y = bounding_box.height * 0.5 + self.scale_y * 0.5

        position = PedestrianTracker.__rectCenter(bounding_box)

        # (Re)activate the tracker if correction count exceeds a certain threshold
        if self.correction_count >= PedestrianTracker.TRACKER_ACTIVATE_COUNT \
                and (self.state == TrackState.INACTIVE or self.state == TrackState.START):
            self.state = TrackState.ACTIVE

            self.__initializeKalmanFilter(bounding_box)
            self.predict()

        measurement = [position.x, position.y]

        # Update the predicted state of the kalman filter using the measurements
        self.filter.correct(measurement)

        # If the tracker is active, update the visual identity
        if self.state != TrackState.INACTIVE:
            blend_alpha = 0.4  # TODO: Parameter ?
            cv2.addWeighted(self.color_data, blend_alpha,
                            feature, 1 - blend_alpha, self.color_data)

    # Returns cost
    # Cost should be between 0 and 1
    # 0 -> means detected object probably is the tracked object
    # 1 -> means it's not
    #TODO: Feature is necessary here?
    def cost(self, bounding_box, feature):
        position = PedestrianTracker.__rectCenter(bounding_box)
        prediction = [self.filter.statePre[0], self.filter.statePre[1]]

        positional_cost = np.linalg.norm(prediction - position)
        positional_cost = 1 if positional_cost > PedestrianTracker.TRACKER_MAX_DISTANCE \
            else positional_cost / PedestrianTracker.TRACKER_MAX_DISTANCE

        # Distance between features in sense of pixels
        similarity_cost = cv2.norm(feature, self.color_data, cv2.NORM_L2)

        if self.state == TrackState.INACTIVE:
            # TODO: Parameters ?
            return similarity_cost * 0.7 + positional_cost * 0.3

        return (positional_cost * PedestrianTracker.TRACKER_SCORE_MIX_RATIO) + \
               (similarity_cost * (1.0 - PedestrianTracker.TRACKER_SCORE_MIX_RATIO))

    # Returns the estimated bounding box
    def getRect(self):
        last_position = [self.filter.statePost[0], self.filter.statePost[1]]
        return (last_position[0] - self.scale_x / 2,
                last_position[1] - self.scale_y / 2,
                self.scale_x, self.scale_y)

    # Returns the estimated velocity of the target
    def getVelocity(self):
        return (self.filter.satePost[2], self.filter.statePost[3])

    # Returns the estimated bounding box
    def __rectCenter(self,rect):
        last_position = (self.filter.statePost[0][0],self.filter.statePost[1][0])
        return (last_position[0] - self.scale_x/2, last_position[1] - self.scale_y/2,
                self.scale_x, self.scale_y)


    def __initializeKalmanFilter(self, initial_bounding_box):
        self.filter = cv2.KalmanFilter(4, 2, 0)
        self.filter.transitionMatrix = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0],
                                        [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

        initial_position = self.__rectCenter(initial_bounding_box)

        # Init Kalman Filter
        self.filter.statePre = np.array([initial_position[0], initial_position[1], 0, 0])
        self.filter.statePost = np.array([initial_position[0], initial_position[1], 0, 0])

        cv2.setIdentity(self.filter.measurementMatrix)
        cv2.setIdentity(self.filter.processNoiseCov, 10 ** -4)
        cv2.setIdentity(self.filter.measurementNoiseCov, 10 ** -1)
        cv2.setIdentity(self.filter.errorCovPost, 10 ** -1)


# TODO: Should write the trackings for each frame in a out.txt
# TODO: Should keep the trackings so far in order to return them afterwards
# TODO: Needs to keep track of major axes and store them as as well
class MultiPedestrianTracker:
    MULTI_TRACKER_ASSOCIATION_THRESHOLD = 0.3

    def __init__(self, detector):
        self.detector = detector
        self.trackers = []
        self.tracking_result = [] #TODO: Keeps the detection results

    def predict(self):
        for tracker in self.trackers:
            tracker.predict()

    def update(self, frame):
        #Get detection ( List of detections as boxes ) Also get the stabilized frame
        stabilized_frame, detections = self.detector.processImage(frame)

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
                            if not tracker_assigned[i]:
                                tracker = self.trackers[j]
                                cost = tracker.cost(detection)

                                if cost < min_cost:
                                    min_cost = cost
                                    best_tracker = tracker
                                    best_detection = detection

                                    best_detection_index = i
                                    best_tracker_index = j

                # Update tracker using detection
                if not best_detection or not best_tracker:
                    break

                # If quality of the best detection/tracker pair is not good enough
                #  Then stop association!!!
                if min_cost > MultiPedestrianTracker.MULTI_TRACKER_ASSOCIATION_THRESHOLD:
                    break

                # Correct tracker using detection
                best_tracker.correct(best_detection)

                # Update assignment stats
                detection_assigned[best_detection_index] = True
                tracker_assigned[best_tracker_index] = True
                num_tracker_assigned += 1
                num_detection_assigned += 1

        # Step 2: Create new trackers for unassigned detections
        #         Maybe create detection only if detection has high confidence

        if num_detections >= 0 and num_detection_assigned < num_detections:
            for i in range(num_detections):
                if not detection_assigned[i]:
                    detection = detections[i]

                    # TODO: Create new trackers, features are optional
                    self.trackers.append(PedestrianTracker(detection))

        # Step 3: Handle unassigned trackers (Remove Dead ones)

        # Remove dead trackers
        self.trackers = list(filter(lambda t: t.state != TrackState.DEAD, self.trackers))

        #Return the stabilized frame
        return stabilized_frame
