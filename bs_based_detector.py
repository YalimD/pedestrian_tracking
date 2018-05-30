import argparse
from datetime import datetime

from detection_tracking_lib import *

import cv2

"""
Created on Tue Apr 24 17:50:19 2018

@author: yalim
@coauthor: serkan
"""

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


class VideoProcessor:

    #TODO: should also contain bs paramteres and pass them to detector
    def __init__(self, detector, confidence):

        #Initialize Video Reader
        if args.source and args.source != '0':
            print("Playing from source {}".format(args.source))
            source = args.source
            if source.isdigit():
                source = int(source)
        else:
            source = 0
            print("Playing from default source")

        self.videoReader = VideoReader(source)

        self.detector = pedestrian_detector.PedestrianDetector(detector, confidence)
        self.tracker = pedestrian_tracker.MultiPedestrianTracker(self.detector)

    def processVideo(self, source, output_name):

        with self.videoReader as cap:

            initiation = datetime.now().timestamp()
            num_of_frames = 0
            prev_time = initiation
            mean_fps = 0

            # Initialize Video Writer
            out_fps = 30 if (source == 0) else cap.get(cv2.CAP_PROP_FPS)

            # WARNING: For .mp4 files, requires ffmpeg (http://www.ffmpeg.org/) installed
            writer = cv2.VideoWriter(output_name, int(cap.get(cv2.CAP_PROP_FOURCC)),
                                     out_fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            if writer.isOpened():
                print("Started writing the output to {}".format(args.output))
            else:
                print("Failed to start writing output to {}".format(args.output))
                raise IOError

            try:
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

                    # frame,_ = self.detector.processImage(frame)

                    #Predict trackers
                    self.tracker.predict()

                    #The tracker needs to ask the detector to return the detections
                    #Tracker returns the stabilized image (stablized before detection in detector)
                    frame = self.tracker.update(frame)
                    #TODO: Draw the tracked pedestrians

                    cv2.putText(frame, "FPS: {:.2f}".format(mean_fps), (10, 15),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (100, 255, 255), 1, cv2.LINE_AA)

                    cv2.imshow('Detection Output', frame)
                    key = cv2.waitKey(5)

                    if key == 27 or key == ord('q'):
                        break

                    writer.write(frame)
            finally:
                if writer:
                    writer.release()

                print("It took {} seconds for the program to process the output the resulting "
                      "video with {} frames (on average {} fps )".format(datetime.now().timestamp() - initiation, num_of_frames, mean_fps))

                #TODO: Return the detections and major axes

if __name__ == "__main__":

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

    #TODO: Side by side comparison of detections
    videoProcessor = VideoProcessor(args.detector,args.confidence)
    videoProcessor.processVideo(args.source,args.output)



