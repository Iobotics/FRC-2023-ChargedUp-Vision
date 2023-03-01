
# Importing Libraries
import robotpy_apriltag
# https://docs.opencv.org/4.x/
import cv2
import numpy as np
import argparse
import time

# starts the live feed from the camera
capture = cv2.VideoCapture(0)

# function that converts the camera feed to grayscale
def convert(capture):
    gray = cv2.cvtColor(capture, cv2.COLOR_RGB2GRAY);
    # returns the live feed from the camera in grayscale
    return gray

# creates a detector that runs a detector on the feed
det = robotpy_apriltag.AprilTagDetector()
# Copying parameters from the calibration script result
fx, fy, cx, cy = (1395.1527042544735, 1391.6280089009383, 639.070620047402, 353.97356161241004)
# creates an ApriltagPoseEstimator to calculate the position, angle, and distance from the apriltag
config = robotpy_apriltag.AprilTagPoseEstimator.Config(6, 1395.1527042544735, 1391.6280089009383, 639.070620047402, 353.97356161241004)
estimator = robotpy_apriltag.AprilTagPoseEstimator(config)

# adds a family of apriltags
tag = det.addFamily("tag16h5")

# convert height and weight values to int
width = int(capture.get(3))
height = int(capture.get(4))

# records the size of the video
size = (width,height)

result = cv2.VideoWriter('recording.avi', cv2.VideoWriter_fourcc(*'JPEG'), 42, size)

while(True):
    t_begin = time.time()
    ret, frame = capture.read()

    if ret == True:

        # converts stream into grayscale
        processed = frame
        t_end = time.time()

        # returns a prebuilt list of python objects
        detectionList = det.detect(convert(frame))

        # for loop that detects the corners and position of the apriltags
        for i in detectionList:

            # making decision_margin lower than 60 to get rid of the white dots
            margin = i.getDecisionMargin()
            print("Decision Margin:", round(margin))
            #print("No apriltags detected...")

            # detect the tag id:
            id = round(i.getId())

            if (margin > 85): #originally 60
                if (id > 0 or id <= 8):
                    print("No apriltags detected...")
                    # identifying the corners of the apriltag
                    corners = i.getCorners([0]*8)
                    # reshape corners into a 4 by 2 matrix
                    order = np.array(corners).reshape((4,2))
                    # prints the corner points
                    #print(order)

                    # prints the tag id to the terminal
                    print("Tag ID #:", id)

                    #cv2.putText(frame, "ID:", (100,100), font, 1, (255,0,0), 2)
                    # shows the class type (numpy.ndarray)
                    #print("Type:", type(order))
                    # shows data type (float64 in this case)
                    #print("dtype: ", order.dtype)

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

                    # AprilTagPoseEstimator
                    poseEstimator = estimator.estimate(i)
                    rotation = estimator.estimate(i).rotation()
                    translation = estimator.estimate(i).translation()
                    print("Rotation: ", rotation)
                    print("Translation: ", translation)
            else:
                    # shows the image with the circles and lines on
                    cv2.imshow( "Image:", frame)
                    # Writes the recorded frames into the file: 'recording.avi'
                    result.write(frame)

        # Press S on keyboard to stop the process
        if cv2.waitKey(1) == ord('e'):
            break
    # Break the loop
    else:
        break

# release video capture and recording
capture.release()
result.release()

# destroys the video
cv2.destroyAllWindows()

print("The video was recorded!")
