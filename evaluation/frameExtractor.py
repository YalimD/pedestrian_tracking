#Written by Yalım Doğan
#Code for converting the given video into image files, in order to CONSISTENTLY annotate them,
#as vatic.js sometimes "jumps" frames which causes wrong and corrupted tracking data.

import cv2
import numpy
import skvideo.io
import argparse
from os.path import basename
import os.path

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

if __name__ == "__main__":

    # Parse inputs
    parser = argparse.ArgumentParser(
        description="Converts the video info image (frame) files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-v", "--video", help="Video to extract the frames from",default = None)

    args = parser.parse_args()
    source = args.video

    if source is None:
        parser.error("Cannot read given video {}".format(source))


    frame_id = 0
    output_folder = "frames_" + os.path.splitext(basename(source))[0]

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print("Processing...")

    #Open the video
    with VideoReader(source) as cap:
        while True:
            success, frame = cap.read()

            if not success:
                if frame_id == 0:
                    print('Cannot read frame from source {}'.format(source))
                else:
                    print("Done")
                break


            #Read and output the frame
            f_name = output_folder + os.sep + str(frame_id)
            # print(f_name)
            cv2.imwrite( f_name + ".jpg", frame)
            frame_id += 1