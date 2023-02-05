import cscore
import ntcore
import socket
import logging
import cv2
import numpy as np
import robotpy_apriltag
import argparse
import time

# function that converts an image to grayscale
def convert(capture):
    gray = cv2.cvtColor(capture, cv2.COLOR_RGB2GRAY)
    # show the image in grayscale
    return gray

def contours(binary_img, min_area=15):
    contour_list, _ = cv2.findContours(binary_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    return [x for x in contour_list if (cv2.contourArea(x) >= min_area)]

def draw(img, contour_list, drawColor):
    width = img.shape[0]
    height = img.shape[1]
    output_img = np.copy(img)
    x_list = []
    y_list = []
    for contour in contour_list:
        cv2.drawContours(output_img, [contour],-1, color = (0, 255, 0), thickness = 2)

        rect = cv2.minAreaRect(contour)
        center, size, angle = rect
        center = np.array(center, dtype=np.int32)

        # cv2.drawContours(output_img, [cv2.boxPoints(rect).astype(int)], -1, color = drawColor, thickness = 2)
        cv2.circle(output_img, center = center, radius = 3, color = drawColor, thickness = -1)

        convexHull = cv2.convexHull(contour)
    
    return output_img

def processing(img): #returns and accepts bgr image
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cube_img = cv2.inRange(hsvimg, (120,100,100),(160,255,255))
    cts = contours(cube_img)
    output_img = draw(img,cts,(255,0,0))

    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cone_img = cv2.inRange(hsvimg, (20,150,150),(40,255,255))
    cts = contours(cone_img)
    output_img = draw(output_img,cts,(255,0,0))

    return output_img

def aprtags(frame,det,estimator):
    processed = convert(frame)
    #cts = contours(processed)
    #with_labels = draw(frame,  cts)
    t_end = time.time()
    detectionList = det.detect(processed)

    # for loop that detects the corners and position of the apriltags
    for i in detectionList:

        # returns a prebuilt list of python objects
        detectionList = det.detect(processed)

        # decision_margin
        # ignore anything lower than 35
        margin = i.getDecisionMargin()
        print("No AprilTag Detected...")
        #print("Decision Margin:", margin)

        if margin > 60:
            # identifying the corners of the apriltag
            corners = i.getCorners([0]*8)
            # reshape corners into a 4 by 2 matrix
            order = np.array(corners).reshape((4,2))
            # prints the corner points
            #print(order)

            # draw lines from the corner points
            cv2.line(processed, order[0,:].astype(int), order[1,:].astype(int), (0,0,255), 3)
            cv2.line(processed, order[1,:].astype(int), order[2,:].astype(int), (0,0,255), 3)
            cv2.line(processed, order[2,:].astype(int), order[3,:].astype(int), (0,0,255), 3)
            cv2.line(processed, order[3,:].astype(int), order[0,:].astype(int), (0,0,255), 3)

            # converts float number to int to plot a circle on the corner
            cv2.circle(processed, (tuple(order[0,:].astype(int))), 8, (255,100,0), 4) # left bottom
            cv2.circle(processed, (tuple(order[1,:].astype(int))), 8, (255,100,0), 4) # right bottom
            cv2.circle(processed, (tuple(order[2,:].astype(int))), 8, (255,100,0), 4) # right top
            cv2.circle(processed, (tuple(order[3,:].astype(int))), 8, (255,100,0), 4) # left top

            #cv2.putText(capture, "ID", tuple(order[1,:].astype(int)), font, 2, (255,200,150), 5, cv2.LINE_AA, False)
            # AprilTagPoseEstimator
            #pose = estimator.estimate(det)
            # Print the estimated pose
            print("Pose Estimator: ", estimator.estimate(i))
            print(estimator.estimate(i).translation())
            print(estimator.estimate(i).rotation())
        else:
            # shows the image with the circles and lines on
            return processed, estimator.estimate(i)
    return processed, estimator.estimate(i)

def main():
    font = cv2.FONT_HERSHEY_SIMPLEX
    #img = cv2.imread('/Users/helenasieh/Desktop/Vision/apriltagrobots_overlay.png')

    print("* Detecting AprilTags...")
    # creates a detector that runs a detector on the image
    det = robotpy_apriltag.AprilTagDetector()

    # Copying parameters from the calibration script result
    fx, fy, cx, cy = (1395.1527042544735, 1391.6280089009383, 639.070620047402, 353.97356161241004)
    # creates an ApriltagPoseEstimator to calculate the position, angle, distance, etc of apriltag
    config = robotpy_apriltag.AprilTagPoseEstimator.Config(6, 1395.1527042544735, 1391.6280089009383, 639.070620047402, 353.97356161241004)
    estimator = robotpy_apriltag.AprilTagPoseEstimator(config)

    # adds a family of apriltags
    tag = det.addFamily("tag16h5")
    
    cameraSelected = 0
    cs = cscore.CameraServer
    cs.enableLogging()

    camera_1 = cscore.UsbCamera(name = 'camera_1', dev = 0)
    camera_1.setResolution(320,240)
    camera_2 = cscore.UsbCamera(name = 'camera_2', dev = 2)
    camera_2.setResolution(320,240)

    cs.addCamera(camera_1)
    cs.addCamera(camera_2)

    # Does not actually send video back to dashboard
    # Creates cvSource object that is used to send frames back
    outputSource = cs.putVideo('feed',320,240)

    #creates cvSink object that can be used to grab frames
    outputSink1 = cs.getVideo(camera=camera_1)
    outputSink2 = cs.getVideo(camera=camera_2)

    
    #pre allocate img for memory
    img1 = np.zeros(shape=(1280, 960, 3), dtype=np.uint8)
    img2 = np.zeros(shape=(1280, 960, 3), dtype=np.uint8)
    nt = ntcore.NetworkTableInstance.getDefault()
    nt.setServerTeam(2438)
    nt.startClient4("temp")
    sd = nt.getTable("SmartDashboard")
    
    while True:
        # Grabs latest frame and sets to img, out returns 0 if error and time if not error
        out, img1 = outputSink1.grabFrame(img1)

        if out == 0:

            print(outputSink1.getError())
            
            continue
        
        img1 = aprtags(img1,det,estimator)
        
        #out, img2 = outputSink2.grabFrame(img2)

        #if out == 0:
        #    print(outputSink2.getError())
            
        #    continue
        
        #img2 = aprtags(img2)
        outputSource.putFrame(img1)




if __name__ == "__main__":

    main()