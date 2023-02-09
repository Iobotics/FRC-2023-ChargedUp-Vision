#import modules
import cscore
import ntcore # when looking at documentation make sure looking at ntcore and not networktables, networktables is a old version
import cv2
import numpy as np

#import detectors
import april_detector
import openvino_detect


# DEPRECATED: old function for color based object detection
def contours(binary_img, min_area=15): 
    contour_list, _ = cv2.findContours(binary_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    return [x for x in contour_list if (cv2.contourArea(x) >= min_area)]

# DEPRECATED: old function for color based object detection
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

# DEPRECATED: old function for color based object detection
def processing(img):
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cube_img = cv2.inRange(hsvimg, (120,100,100),(160,255,255))
    cts = contours(cube_img)
    output_img = draw(img,cts,(255,0,0))

    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cone_img = cv2.inRange(hsvimg, (20,150,150),(40,255,255))
    cts = contours(cone_img)
    output_img = draw(output_img,cts,(255,0,0))

    return output_img



def main():    
    #initialise network tables instance
    nt = ntcore.NetworkTableInstance.getDefault() # get global network tables instance
    nt.startClient4("2438") 
    nt.setServerTeam(2438) # sets server to send to common ip for 2438

    sd = nt.getTable("SmartDashboard") # send info to smartdashboard table

    cameraSelectionTopic = sd.getIntegerTopic("CameraSelection")
    cameraSelectionSubscriber = cameraSelectionTopic.subscribe(0)

    cs = cscore.CameraServer # set cs to CameraServer singleton object
    cs.enableLogging()

    camera_1 = cscore.UsbCamera(name = 'camera_1', dev = 0) # USB camera 
    camera_1.setResolution(320,240)
    camera_2 = cscore.UsbCamera(name = 'camera_2', dev = 2) # USB camera
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
    aprimg1 = np.zeros(shape=(1280, 960, 3), dtype=np.uint8)
    aprimg2 = np.zeros(shape=(1280, 960, 3), dtype=np.uint8)
    yoloimg1 = np.zeros(shape=(1280, 960, 3), dtype=np.uint8)
    yoloimg2 = np.zeros(shape=(1280, 960, 3), dtype=np.uint8)

    # img = np.zeros(shape=(6, 1280, 960, 3), dtype=np.uint8)

    # creates topics for all info that needs to be sent back
    translationX = sd.getDoubleTopic("TranslationX")
    translationY = sd.getDoubleTopic("TranslationY")
    translationZ = sd.getDoubleTopic("TranslationZ")
    rotationX = sd.getDoubleTopic("RotationX")
    rotationY = sd.getDoubleTopic("RotationY")
    rotationZ = sd.getDoubleTopic("RotationZ")
    tagID = sd.getIntegerTopic("aprilTagID")
    
    # create publishers for all info topics
    translationXPublisher = translationX.publish()
    translationYPublisher = translationY.publish()
    translationZPublisher = translationZ.publish()
    rotationXPublisher = rotationX.publish()
    rotationYPublisher = rotationY.publish()
    rotationZPublisher = rotationZ.publish()
    tagIDPublisher = tagID.publish()
    

    april = april_detector.AprilDetector((320,240))
    yolo = openvino_detect.YoloOpenVinoDetector("weights/")
    while True:
        # Grabs latest frame and sets to img, out returns 0 if error and time if not error
        out, img1 = outputSink1.grabFrame(img1) # img1 in BGR format
        if out == 0: # skips if cant grab frame
            print(outputSink1.getError())
            continue

        out, aprimg1 = april.detect(cv2.resize(img1, (320,240))) # runs apriltag detector, out is list of all important outputs
        if out != 0: 
            print(out)
            translationXPublisher.set(value = out[0])
            translationYPublisher.set(value = out[1])
            translationZPublisher.set(value = out[2])
            rotationXPublisher.set(value = out[3])
            rotationYPublisher.set(value = out[4])
            rotationZPublisher.set(value = out[5])
            tagIDPublisher.set(value = out[6])
        
        out = yolo.detect(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
        yoloimg1 = img1.copy()
        if out != 0:
            for i in out:
                if i["confidence"] < 0.8:
                    continue
                order = i["corners"]

                cv2.line(yoloimg1, order[0,:].astype(int), order[1,:].astype(int), i["color"], 3)
                cv2.line(yoloimg1, order[1,:].astype(int), order[2,:].astype(int), i["color"], 3)
                cv2.line(yoloimg1, order[2,:].astype(int), order[3,:].astype(int), i["color"], 3)
                cv2.line(yoloimg1, order[3,:].astype(int), order[0,:].astype(int), i["color"], 3)

                # converts float number to int to plot a circle on the corner
                cv2.circle(yoloimg1, (tuple(order[0,:].astype(int))), 8, i["color"], 4) # left bottom
                cv2.circle(yoloimg1, (tuple(order[1,:].astype(int))), 8, i["color"], 4) # right bottom
                cv2.circle(yoloimg1, (tuple(order[2,:].astype(int))), 8, i["color"], 4) # right top
                cv2.circle(yoloimg1, (tuple(order[3,:].astype(int))), 8, i["color"], 4) # left top


        # cv2.imshow('f',yoloimg1)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        cameraSelected = cameraSelectionSubscriber.get()
        # outputSource.putFrame(img1 if cameraSelected == 0 else img2 if cameraSelected == 1 else aprimg if cameraSelected == 2 else yoloimg)
        outputSource.putFrame(aprimg1)




if __name__ == "__main__":
    main()
