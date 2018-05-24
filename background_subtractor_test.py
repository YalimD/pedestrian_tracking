from pedestrian_tracker.bgsegm import *
from pedestrian_tracker.stab import *

import numpy as np
import cv2


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


with VideoReader(0) as cap:
    bs_with_stab = BackgroundSubtractor(method='mog', stabilizer='lk')
    bs_without_stab = BackgroundSubtractor(method='mog', stabilizer=None)

    while True:
        success, orig_frame = cap.read()

        mask_with_stab, bs_frame_with_stab = bs_with_stab.apply(orig_frame)
        bs_frame_with_stab[mask_with_stab > 0, 0] = 255

        mask, bs_frame = bs_without_stab.apply(orig_frame)
        bs_frame[mask > 0, 0] = 255

        merged_frame = np.zeros((mask.shape[0], mask.shape[1] * 2, 3),dtype=np.uint8)

        merged_frame[:, 0:mask.shape[1], :] = bs_frame
        merged_frame[:, mask.shape[1]:, :] = bs_frame_with_stab
        cv2.imshow('Display', merged_frame)

        key = cv2.waitKey(1)

        if key == 27 or key == ord('q'):
            break