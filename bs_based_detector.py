import argparse
import os
from datetime import datetime
from os.path import basename

import cv2

from detection_tracking_lib import *

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

    # TODO: should also contain bs parameters and pass them to detector
    def __init__(self, detector, confidence, detector_output, remove_shadows=False, stabilize=True):

        self.detector = pedestrian_detector.PedestrianDetector(detector, confidence= confidence,
                                                               det_out_name = detector_output, stabilize = stabilize)
        self.tracker = pedestrian_tracker.MultiPedestrianTracker(self.detector, remove_shadows)

    def process_video(self, source, results_folder, text_output_name, video_output_name, use_sk_writer=False):

        # Initialize Video Reader
        if source and source != '0':
            source_name = os.path.splitext(basename(source))[0]
            print("Playing from file {}".format(source_name))
            if source.isdigit():
                source = int(source)
        else:
            source = 0
            source_name = "camera"
            print("Playing from default source")

        with VideoReader(source) as cap:

            initiation = datetime.now().timestamp()
            frameNum = 0
            prev_time = initiation
            mean_fps = 0

            if len(results_folder) == 0:
                results_folder = os.path.join(os.path.dirname(source), "results_{}".format(source_name))

            if not os.path.exists(results_folder):
                os.mkdir(results_folder)

            # Open detector file
            self.detector.openFile(results_folder)

            text_output_name = results_folder + text_output_name
            # Initialize the output text file
            print("Writing to {}".format(text_output_name))
            text_out = open(text_output_name, "w")

            outputing_video = len(video_output_name) > 0
            if outputing_video:

                posture_v_name = results_folder + "postures_" + video_output_name
                det_v_name = results_folder + "detected_" + video_output_name
                stabilzed_v_name = results_folder + "stabilized_" + video_output_name

                if not use_sk_writer:
                    # Initialize Video Writer (s)
                    out_fps = 30 if (source == 0) else cap.get(cv2.CAP_PROP_FPS)

                    # WARNING: For .mp4 files, requires ffmpeg (http://www.ffmpeg.org/) installed
                    posture_videowriter = cv2.VideoWriter(posture_v_name, int(cap.get(cv2.CAP_PROP_FOURCC)),
                                             out_fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                    detection_videowriter = cv2.VideoWriter(det_v_name, int(cap.get(cv2.CAP_PROP_FOURCC)),
                                                            out_fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                    stabilized_videowriter = cv2.VideoWriter(stabilzed_v_name, int(cap.get(cv2.CAP_PROP_FOURCC)),
                                             out_fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

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

                        #The writer has no attribute to check if it is created successfully
                        def isOpened(self):
                            return True

                    detection_videowriter = VideoWriterWrapper(det_v_name)
                    stabilized_videowriter = VideoWriterWrapper(stabilzed_v_name)
                    posture_videowriter = VideoWriterWrapper(posture_v_name)

                if detection_videowriter.isOpened() and stabilized_videowriter.isOpened():
                    print("Started writing the postures to {}, output with detections only to {}"
                          " and stabilized only result to {}".format(posture_v_name,det_v_name, stabilzed_v_name))
                else:
                    print("Failed to start writing video outputs")
                    raise IOError
            try:
                while True:

                    # Time calculation
                    cur_time = datetime.now().timestamp()
                    time_elapsed = cur_time - prev_time  # In seconds
                    cur_fps = 1 / max(0.001, time_elapsed)
                    mean_fps = mean_fps * 0.5 + cur_fps * 0.5
                    prev_time = cur_time

                    print("Mean FPS: {} -  Frames so far: {}".format(mean_fps, frameNum))
                    success, frame = cap.read()  # Get frame from video capture
                    frameNum += 1

                    # Bail out if we cannot read the frame
                    if not success:
                        print('Cannot read frame from source {}'.format(source))
                        break

                    # Predict trackers
                    self.tracker.predict()

                    # The tracker needs to ask the detector to return the detections
                    # Tracker returns the stabilized image (stablized before detection in detector)
                    frame = self.tracker.update(frame, frameNum)

                    #Write to stabilized only video before drawing the pedestrians on it
                    if outputing_video:
                        stabilized_videowriter.write(frame)

                    posture_frame = self.tracker.draw_and_write_trackers(frame, frameNum, text_out)

                    cv2.putText(frame, "FPS: {:.2f}".format(mean_fps), (10, 15),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (100, 255, 255), 1, cv2.LINE_AA)

                    #TODO: Flag
                    # cv2.imshow('Detection Output', frame)
                    key = cv2.waitKey(1)

                    if key == 27 or key == ord('q'):
                        break

                    if outputing_video:
                        posture_videowriter.write(posture_frame)
                        detection_videowriter.write(frame)
            except KeyboardInterrupt:
                print("Interrupted, saving videos...")
            finally:
                if outputing_video:
                    detection_videowriter.release()
                    stabilized_videowriter.release()

                # Also write the diagnostics to the end of the detections file
                diag = "It took {} seconds for the program to process the output the resulting video with {} frames" \
                       " with confidence (if applies) {}" \
                      "(on average {} fps)".format(datetime.now().timestamp() - initiation, frameNum,
                                                   self.detector.confidence, mean_fps)
                self.detector.det_output.write(diag)

                text_out.close()
                self.detector.closeFile()


if __name__ == "__main__":

    # Parse inputs
    parser = argparse.ArgumentParser(
        description="Runs model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-s', '--source', help="Source of the video", default='0')
    parser.add_argument('-o', '--output_folder', help="Output folder for the results", default='')
    parser.add_argument('--v_output', help='Name of the output video (with tracking)', default='')
    parser.add_argument('--d_output', help='Name of the output text file of detections only', default='det_out.txt')
    parser.add_argument('--t_output', help='Name of the output text file of trackers', default='track_out.txt')
    parser.add_argument('-d', '--detector',
                        help="The detector to be used (if rnn, pass the folder containing the related"
                             "graph files)", default='hog')
    parser.add_argument('-c', '--confidence', help="Detection confidence", type=float, default=0.35)
    parser.add_argument('--remShad', help="Remove Shadows", const=True,
                        default=False, nargs='?')
    parser.add_argument('--useSkImage', help="Use skimage Video Writer", const=True,
                        default=False, nargs='?')
    parser.add_argument('--stab', help="Stabilize the image", const=True,
                        default=False, nargs='?')
    # TODO: Background subtraction, hog parameters needs to be added

    args = parser.parse_args()

    videoProcessor = VideoProcessor(args.detector, args.confidence, args.d_output, args.remShad, args.stab)
    videoProcessor.process_video(args.source, args.output_folder, args.t_output, args.v_output, args.useSkImage)
