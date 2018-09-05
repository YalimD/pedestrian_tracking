"""Provides Background-Forgeround segmentation routines
"""

import numpy as np
import cv2
from background_subtractor.stab import *

__all__ = ['BackgroundSubtractor']

class BackgroundSubtractor:
    """
    BackgroundSubtractor encapsulates various background subtraction methods

    Attributes
    ----------
    _background_subtractor: object
        Internal background subtractor object.

    _stabilizer: VideoStablizer
        Video stabilization method

    """

    def __init__(self, method, method_args={}, stabilize = True, stabilizer=None, stabilizer_args={}):
        """
        Parameters
        ----------
        method: { 'mog', 'lsbp', 'gsoc', 'cnt', 'gmg'}, optional
            Determines which background subtraction is used

        method_args: dict, optional
            Method specific arguments

        stabilize: True if the frame needs to be stabilized before background subtraction

        stabilizer: VideoStablizer or string, optional
            Which stabilizer to use (see also: stab module)


        stabilizer_args: dict, optional
            Stabilizer specific arguments. Only used if stabilizer is a string

        """

        # Initialize background subtractor method
        background_subtraction_methods = {
            'mog': cv2.bgsegm.createBackgroundSubtractorMOG,
            'lsbp': cv2.bgsegm.createBackgroundSubtractorLSBP,
            'gsoc': cv2.bgsegm.createBackgroundSubtractorGSOC,
            'cnt': cv2.bgsegm.createBackgroundSubtractorCNT,
            'gmg': cv2.bgsegm.createBackgroundSubtractorGMG
        }

        if method.lower() in background_subtraction_methods:
            self._subtractor = background_subtraction_methods[method.lower()](**method_args)
        else:
            type_names = str(list(background_subtraction_methods.keys()))
            raise ValueError('Invalid background subtractor method. Method must be one of : {}'.format(type_names))

        if isinstance(stabilizer, str):
            self._stabilizer = VideoStablizer(stabilizer, **stabilizer_args)
        else:
            self._stabilizer = stabilizer

        self.stabilize = stabilize
        self._prev_mask = None

    def apply(self, image):
        """Applies the background subtraction. Returns mask

        Parameters
        ----------
        image: array_like
            Input frame

        Returns
        -------
        mask: nd_array
            Foreground-Background mask matrix. In mask matrix a value is 255 if it is a foreground.

        frame: nd_array
            Image that background subtraction is applied.
            If stabilizer is not defined this frame is same as input frame

        """

        image = np.asarray(image)

        if self._stabilizer is not None and self.stabilize is True:
            image = self._stabilizer.stabilize(image, mask=self._prev_mask)

        mask = self._subtractor.apply(image)

        self._prev_mask = mask
        return mask, image