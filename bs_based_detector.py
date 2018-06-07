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

    # TODO: should also contain bs paramteres and pass them to detector
    def __init__(self, detector, confidence, removeShadows=False):

        self.detector = pedestrian_detector.PedestrianDetector(detector, confidence)
        self.tracker = pedestrian_tracker.MultiPedestrianTracker(self.detector, removeShadows)

    def processVideo(self, source, text_output_name, video_output_name, use_cv_writer=False):

        # Initialize Video Reader
        if source and source != '0':
            print("Playing from source {}".format(source))
            if source.isdigit():
                source = int(source)
        else:
            source = 0
            print("Playing from default source")

        with VideoReader(source) as cap:

            initiation = datetime.now().timestamp()
            frameNum = 0
            prev_time = initiation
            mean_fps = 0

            print("Writing to {}".format(text_output_name))
            # Initialize the output text file
            open(text_output_name, "w").close()
            text_out = open(text_output_name, "a")
            text_out.write("{} {} {} {} {} \n".format(source,
                                                      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                                      1,
                                                      cap.get(cv2.CAP_PROP_FPS)
                                                      ))

            outputing_video = len(video_output_name) > 0
            if outputing_video:

                if use_cv_writer:
                    # Initialize Video Writer
                    out_fps = 30 if (source == 0) else cap.get(cv2.CAP_PROP_FPS)

                    # WARNING: For .mp4 files, requires ffmpeg (http://www.ffmpeg.org/) installed
                    writer = cv2.VideoWriter(video_output_name, int(cap.get(cv2.CAP_PROP_FOURCC)),
                                             out_fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

                    if writer.isOpened():
                        print("Started writing the output to {}".format(args.output))
                    else:
                        print("Failed to start writing output to {}".format(args.output))
                        raise IOError
                else:
                    import skvideo.io
                    class VideoWriterWrapper:
                        def __init__(self, fname):
                            self.fname = fname
                            self.writer = skvideo.io.FFmpegWriter(fname)

                        def write(self, frame):
                            self.writer.writeFrame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                        def release(self):
                            self.writer.close()

                    writer = VideoWriterWrapper(video_output_name)
            try:
                while True:

                    # Time calculation
                    cur_time = datetime.now().timestamp()
                    time_elapsed = cur_time - prev_time  # In seconds
                    cur_fps = 1 / max(0.001, time_elapsed)
                    mean_fps = mean_fps * 0.5 + cur_fps * 0.5
                    prev_time = cur_time

                    print("Mean FPS: {}".format(mean_fps))
                    success, frame = cap.read()  # Get frame from video capture
                    frameNum += 1

                    # Bail out if we cannot read the frame
                    if not success:
                        print('Cannot read frame from source {}'.format(source))
                        break

                    # frame,_ = self.detector.processImage(frame)

                    # Predict trackers
                    self.tracker.predict()

                    # The tracker needs to ask the detector to return the detections
                    # Tracker returns the stabilized image (stablized before detection in detector)
                    frame = self.tracker.update(frame)

                    text_out.write("{},".format(frameNum))
                    self.tracker.draw_and_write_trackers(frame, text_out)
                    text_out.write("\n")

                    cv2.putText(frame, "FPS: {:.2f}".format(mean_fps), (10, 15),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (100, 255, 255), 1, cv2.LINE_AA)

                    cv2.imshow('Detection Output', frame)
                    key = cv2.waitKey(5)

                    if key == 27 or key == ord('q'):
                        break

                    if outputing_video:
                        writer.write(frame)
            finally:
                if outputing_video and writer:
                    writer.release()

                text_out.close()

                print("It took {} seconds for the program to process the output the resulting video with {} frames " \
                      "(on average {} fps )".format(datetime.now().timestamp() - initiation, frameNum, mean_fps))

                # TODO: Return the detections and major axes


if __name__ == "__main__":
    # Parse inputs
    parser = argparse.ArgumentParser(
        description="Runs model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-s', '--source', help="Source of the video", default='0')
    parser.add_argument('--v_output', help='Name of the output video (with tracking)', default='')
    parser.add_argument('--t_output', help='Name of the output text file of trackers', default='out.txt')
    parser.add_argument('-d', '--detector',
                        help="The detector to be used (if rnn, pass the folder containing the related"
                             "graph files)", default='hog')
    parser.add_argument('-c', '--confidence', help="Detection confidence", type=float, default=0.2)
    parser.add_argument('--remShad', help="Remove Shadows", const=True,
                        default=False, nargs='?')
    parser.add_argument('--useCvWriter', help="Use opencv Video Writer", const=True,
                        default=False, nargs='?')
    # TODO: Background subtraction, stabilization and hog parameters needs to be added

    args = parser.parse_args()

    videoProcessor = VideoProcessor(args.detector, args.confidence, args.remShad)
    videoProcessor.processVideo(args.source, args.t_output, args.v_output, use_cv_writer=args.useCvWriter)
