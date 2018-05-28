"""
"""

import numpy as np
import cv2

__all__ = ['VideoStablizer']


class LKVideoStabilizer:
    """
    Lucas-Kanade based video stabilizer
    An explanation: http://concept-script.com/flowpoints/page01.html
    TODO: Documentation
    """

    def __init__(self, num_points_to_track=100):
        self.old_pyramid = None
        self.feature_params = dict(maxCorners=num_points_to_track, qualityLevel=0.01,
                                   minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.max_forward_backward_error = 0.3

    def stabilize(self, image, old_segm_mask=None):
        image = np.asarray(image)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # FIXME: buildOpticalFlowPyramid does not work on opencv 3.4.1. Bug is already reported.
        #  see https://github.com/opencv/opencv/issues/8268

        # Build tracking pyramid
        # _, new_pyramid = cv2.buildOpticalFlowPyramid(gray_image,
        #                                             winSize=self.lk_params['winSize'],
        #                                             maxLevel=self.lk_params['maxLevel'])
        new_pyramid = gray_image

        # If this is the first run
        if self.old_pyramid is None:
            self.old_pyramid = new_pyramid
            self.old_image = image
            return image
        else:
            # Find 'good' points to track :P
            if old_segm_mask is None:
                old_points = cv2.goodFeaturesToTrack(self.old_pyramid, **self.feature_params)
            else:
                old_segm_mask = ((old_segm_mask == 0) * 255).astype(np.uint8)
                old_points = cv2.goodFeaturesToTrack(self.old_pyramid, mask=old_segm_mask, **self.feature_params)

            if old_points is None:
                raise RuntimeError('Stabilizer cannot stabilize video. No feature points found')

            # Forward - Backward validation
            # First find next points using old points, then use new points to find old_points backwards
            #  If the distance between original old point and tracked old point is greater than a value
            #  Remove it from track list
            new_points, status_forward, _ = cv2.calcOpticalFlowPyrLK(self.old_pyramid, new_pyramid, old_points, None,
                                                                     **self.lk_params)
            old_points_backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(new_pyramid, self.old_pyramid,
                                                                               new_points, None, **self.lk_params)

            # select valid tracks
            forward_backward_error = np.sqrt(
                np.sum(np.square(old_points[:, 0, :] - old_points_backward[:, 0, :]), axis=1))  # forward-backward error
            status = (status_forward.ravel() == 1) & (status_backward.ravel() == 1) & (
                        forward_backward_error < self.max_forward_backward_error)

            old_points = old_points[status == 1]
            new_points = new_points[status == 1]

            # Find old_image -> new_image homography
            homo, _ = cv2.findHomography(old_points[:, 0, :], new_points[:, 0, :], cv2.RANSAC,5.0)

            stabilized_image = image

            if homo is not None:
                mask = cv2.warpPerspective(np.zeros_like(stabilized_image), homo,
                                           (stabilized_image.shape[1], stabilized_image.shape[0]),
                                           flags=cv2.WARP_INVERSE_MAP, borderValue=(255, 255, 255))

                stabilized_image = cv2.warpPerspective(stabilized_image, homo,
                                                       (stabilized_image.shape[1], stabilized_image.shape[0]),
                                                       flags=cv2.WARP_INVERSE_MAP, borderValue=(0, 0, 0))

                stabilized_image[mask > 0] = self.old_image[mask > 0]
                pass

            # FIXME: Pyramid bug
            self.old_pyramid = cv2.cvtColor(stabilized_image, cv2.COLOR_BGR2GRAY)
            self.old_image = stabilized_image
            return stabilized_image

        pass


class VideoStablizer:
    """
    Encapsulates various video stabilization methods
    """

    def __init__(self, method, method_args={}):
        methods = {
            'lk': LKVideoStabilizer
        }

        if method.lower() in methods:
            self._stabilizer = methods[method.lower()](**method_args)

    def stabilize(self, image, mask=None):
        return self._stabilizer.stabilize(image, old_segm_mask=mask)
