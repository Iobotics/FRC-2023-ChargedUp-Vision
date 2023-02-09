# Class to read april tags in image
import cv2
import time
import robotpy_apriltag
import numpy as np

class AprilDetector:
    def __init__(self, size):
        # creates a detector that runs a detector on the image

        cfg = robotpy_apriltag.AprilTagDetector().Config()
        cfg.debug = False
        cfg.decodeSharpening = 0.25
        cfg.numThreads = 4
        cfg.quadDecimate = 2.0
        cfg.quadSigma = 0.0
        cfg.refineEdges = True

        self.det = robotpy_apriltag.AprilTagDetector()
        self.det.setConfig(cfg)

        # Copying parameters from the calibration script result
        fx, fy, cx, cy = (1395.1527042544735, 1391.6280089009383, 639.070620047402, 353.97356161241004)
        # creates an ApriltagPoseEstimator to calculate the position, angle, distance, etc of apriltag
        config = robotpy_apriltag.AprilTagPoseEstimator.Config(0.153*(size[0]/1280) , fx, fy, cx, cy)
        self.estimator = robotpy_apriltag.AprilTagPoseEstimator(config)

        # adds a family of apriltags
        tag = self.det.addFamily("tag16h5")

    def convert(self,frame): # converst image from RGB to Grayscale
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    def detect(self,frame):
        processed = self.convert(frame)
        t_end = time.time()
        detectionList = self.det.detect(processed)

        # for loop that detects the corners and position of the apriltags
        for i in detectionList:

            # returns a prebuilt list of python objects
            detectionList = self.det.detect(processed)

            # decision_margin
            # ignore anything lower than 35
            margin = i.getDecisionMargin()

            if margin > 60:
                # identifying the corners of the apriltag
                corners = i.getCorners([0]*8)
                # reshape corners into a 4 by 2 matrix
                order = np.array(corners).reshape((4,2))

                # draw lines from the corner points
                cv2.line(frame, order[0,:].astype(int), order[1,:].astype(int), (0,0,255), 3)
                cv2.line(frame, order[1,:].astype(int), order[2,:].astype(int), (0,0,255), 3)
                cv2.line(frame, order[2,:].astype(int), order[3,:].astype(int), (0,0,255), 3)
                cv2.line(frame, order[3,:].astype(int), order[0,:].astype(int), (0,0,255), 3)

                # converts float number to int to plot a circle on the corner
                cv2.circle(frame, (tuple(order[0,:].astype(int))), 8, (255,100,0), 4) # left bottom
                cv2.circle(frame, (tuple(order[1,:].astype(int))), 8, (255,100,0), 4) # right bottom
                cv2.circle(frame, (tuple(order[2,:].astype(int))), 8, (255,100,0), 4) # right top
                cv2.circle(frame, (tuple(order[3,:].astype(int))), 8, (255,100,0), 4) # left top

                #cv2.putText(capture, "ID", tuple(order[1,:].astype(int)), font, 2, (255,200,150), 5, cv2.LINE_AA, False)
                # AprilTagPoseEstimator
                #pose = estimator.estimate(det)
                translate = self.estimator.estimate(i).translation()
                rotation = self.estimator.estimate(i).rotation()
                return [translate.X(),translate.Y(),translate.Z(), rotation.X(),rotation.Y(),rotation.Z(), i.getID()], frame
            else:
                # shows the image with the circles and lines on
                return 0, frame
        return 0, frame
