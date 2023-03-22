#import modules
import cscore
import ntcore # when looking at documentation make sure looking at ntcore and not networktables, networktables is a old version
import cv2
import numpy as np
import time
import os

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

def getArea(corners):
    return (corners[2][0]-corners[0][0])*(corners[2][1]-corners[0][1])

def main():
    OUT_RESOLUTION = 320,240
    APRIL_RESOLUTION = 320,240
    CAMERA_RESOLUTION = 320,240

    #initialise network tables instance
    nt = ntcore.NetworkTableInstance.getDefault() # get global network tables instance
    nt.startClient4("2438") 
    nt.setServerTeam(2438) # sets server to send to common ip for 2438

    sd = nt.getTable("SmartDashboard") # send info to smartdashboard table

    cameraSelectionTopic = sd.getIntegerTopic("Camera")
    cameraSelectionSubscriber = cameraSelectionTopic.subscribe(defaultValue = 0)

    cs = cscore.CameraServer # set cs to CameraServer singleton object
    cs.enableLogging()

    camera_1 = cscore.UsbCamera(name = 'camera_1', dev = 0) # USB camera 
    camera_1.setResolution(CAMERA_RESOLUTION[0],CAMERA_RESOLUTION[1])
    camera_2 = cscore.UsbCamera(name = 'camera_2', dev = 2) # USB camera
    camera_2.setResolution(CAMERA_RESOLUTION[0],CAMERA_RESOLUTION[1])

    cs.addCamera(camera_1)
    cs.addCamera(camera_2)

    # Does not actually send video back to dashboard
    # Creates cvSource object that is used to send frames back
    outputSource = cs.putVideo('feed',1280,960)

    #creates cvSink object that can be used to grab frames
    outputSink1 = cs.getVideo(camera=camera_1)
    outputSink2 = cs.getVideo(camera=camera_2)

    
    #pre allocate img for memory
    raw1 = np.zeros(shape=(960, 1280, 3), dtype=np.uint8)
    raw2 = np.zeros(shape=(960, 1280, 3), dtype=np.uint8)
    processed1 = np.zeros(shape=(960, 1280, 3), dtype=np.uint8)
    processed2 = np.zeros(shape=(960, 1280, 3), dtype=np.uint8)

    # img = np.zeros(shape=(6, 1280, 960, 3), dtype=np.uint8)

    # creates topics for all info that needs to be sent back
    translationX = sd.getDoubleTopic("TranslationX")
    translationY = sd.getDoubleTopic("TranslationY")
    translationZ = sd.getDoubleTopic("TranslationZ")
    rotationX = sd.getDoubleTopic("RotationX")
    rotationY = sd.getDoubleTopic("RotationY")
    rotationZ = sd.getDoubleTopic("RotationZ")
    tagID = sd.getIntegerTopic("aprilTagID")
    tagCenterX = sd.getDoubleTopic("tagCenterX")
    tagCenterY = sd.getDoubleTopic("tagCenterY")

    objectInFrame = sd.getBooleanTopic("objectInFrame")
    objectX = sd.getDoubleTopic("objectX")
    objectId = sd.getIntegerTopic("objectId")
    
    # create publishers for all info topics
    translationXPublisher = translationX.publish()
    translationYPublisher = translationY.publish()
    translationZPublisher = translationZ.publish()
    rotationXPublisher = rotationX.publish()
    rotationYPublisher = rotationY.publish()
    rotationZPublisher = rotationZ.publish()
    tagIDPublisher = tagID.publish()
    tagCenterXPublisher = tagCenterX.publish()
    tagCenterYPublisher = tagCenterY.publish()

    objectInFramePublisher = objectInFrame.publish()
    objectXPublisher = objectX.publish()
    objectIdPublisher = objectId.publish()
    

    april = april_detector.AprilDetector(APRIL_RESOLUTION)
    yolo = openvino_detect.YoloOpenVinoDetector("/home/team2438/FRC-2023-ChargedUp-Vision/best_openvino_model/")
    framerate = []
    april_percent = []
    object_percent = []

    i = 0
    while os.path.isfile("raw1-"+str(i)+".avi"):
        i+=1
    videopath1 = "raw1-"+str(i)+".avi"
    raw1Video = cv2.VideoWriter(videopath1, cv2.VideoWriter_fourcc(*'MJPG'), 25, CAMERA_RESOLUTION)

    i = 0
    while os.path.isfile("raw2-"+str(i)+".avi"):
        i+=1
    videopath2 = "raw2-"+str(i)+".avi"
    raw2Video = cv2.VideoWriter(videopath2, cv2.VideoWriter_fourcc(*'MJPG'), 25, CAMERA_RESOLUTION)

    i = 0
    while os.path.isfile("processed1-"+str(i)+".avi"):
        i+=1
    videopath1 = "processed1-"+str(i)+".avi"
    processed1Video = cv2.VideoWriter(videopath1, cv2.VideoWriter_fourcc(*'MJPG'), 25, CAMERA_RESOLUTION)

    i = 0
    while os.path.isfile("processed2-"+str(i)+".avi"):
        i+=1
    videopath2 = "processed2-"+str(i)+".avi"
    processed2Video = cv2.VideoWriter(videopath2, cv2.VideoWriter_fourcc(*'MJPG'), 25, CAMERA_RESOLUTION)


    while True:
        t_0 = time.time()
        # Grabs latest frame and sets to img, out returns 0 if error and time if not error
        out, raw1 = outputSink1.grabFrame(raw1) # raw1 in BGR format
        if out == 0: # skips if cant grab frame
            print("Camera source 1 returned: ", outputSink1.getError())
            continue

        raw1Video.write(raw1)

        out, raw2 = outputSink2.grabFrame(raw2) # raw1 in BGR format
        if out == 0: # skips if cant grab frame
            print("Camera source 2 returned: ", outputSink2.getError())
            continue

        raw2Video.write(raw2)

        out = 0

        processed2 = raw2.copy()

        april_time_0 = time.time()
        out, processed1 = april.detect(cv2.resize(raw1, APRIL_RESOLUTION)) # runs apriltag detector, out is list of all important outputs
        processed1 = cv2.resize(processed1, OUT_RESOLUTION)
        if out == 0: 
            translationXPublisher.set(value = 0)
            translationYPublisher.set(value = 0)
            translationZPublisher.set(value = 0)
            rotationXPublisher.set(value = 0)
            rotationYPublisher.set(value = 0)
            rotationZPublisher.set(value = 0)
            tagIDPublisher.set(value = 43)
        else:
            translationXPublisher.set(value = out[0])
            translationYPublisher.set(value = out[1])
            translationZPublisher.set(value = out[2])
            rotationXPublisher.set(value = out[3])
            rotationYPublisher.set(value = out[4])
            rotationZPublisher.set(value = out[5])
            tagIDPublisher.set(value = out[6])

            # Normalize the apriltag center position to (-1.0, 1.0)
            half_w = APRIL_RESOLUTION[0] // 2
            half_h = APRIL_RESOLUTION[1] // 2
            norm_x = (out[7] - half_w) / half_w
            norm_y = (out[8] - half_h) / half_h
            tagCenterXPublisher.set(value = norm_x)
            tagCenterYPublisher.set(value = norm_y)


        april_time_1 = time.time()
        
        out = 0

        obj_time_0 = time.time()

        cameraSelected = cameraSelectionSubscriber.get(defaultValue=0) % 2

        # yoloraw1 = cv2.resize(raw1, OUT_RESOLUTION)
        out = yolo.detect(cv2.cvtColor(cv2.resize(raw1 if cameraSelected == 0 else raw2, OUT_RESOLUTION),cv2.COLOR_BGR2RGB))
        if out != []:

            largest = -1

            for i in range(len(out)):
            
                if out[i]["confidence"] > 0.5:

                    if getArea(out[i]["corners"]) > (getArea(out[largest]["corners"]) if largest != -1 else 0):

                        largest = i

            if largest != -1:
                order = out[largest]["corners"]

                img = processed1 if cameraSelected == 0 else processed2

                cv2.line(img, order[0,:].astype(int), order[1,:].astype(int), out[largest]["color"], 3)
                cv2.line(img, order[1,:].astype(int), order[2,:].astype(int), out[largest]["color"], 3)
                cv2.line(img, order[2,:].astype(int), order[3,:].astype(int), out[largest]["color"], 3)
                cv2.line(img, order[3,:].astype(int), order[0,:].astype(int), out[largest]["color"], 3)

                cv2.putText(img,"id: "+str(out[largest]["id"])+", "+str(out[largest]["confidence"]),order[0,:].astype(int),cv2.FONT_HERSHEY_SIMPLEX,1,out[largest]["color"],2)

                objectInFramePublisher.set(True)
                objectXPublisher.set((order[0][0]+order[2][0])/320 - 1)
                objectIdPublisher.set(out[largest]["id"])
        else:
            objectInFramePublisher.set(False)
            objectXPublisher.set(0)
            objectIdPublisher.set(-1)

        obj_time_1 = time.time()
        t_1 = time.time()
        # print(1 / (t_1 - t_0), "iters/sec")
        framerate.append(1 / (t_1 - t_0))
        april_percent.append((april_time_1-april_time_0)/(t_1-t_0)*100)
        object_percent.append((obj_time_1-obj_time_0)/(t_1-t_0)*100)


        # print(sum(framerate) / len(framerate), "framerate")
        # print(sum(april_percent)/ len(april_percent),"april percentage")
        # print(sum(object_percent)/ len(object_percent), "object percentage")

        # cv2.imshow('f',processed1)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break

        processed1Video.write(processed1)
        processed2Video.write(processed2)

        cameraSelected = cameraSelectionSubscriber.get(defaultValue=0) % 2
        outputSource.putFrame(processed1 if cameraSelected == 0 else processed2)
        # outputSource.putFrame(raw2)




if __name__ == "__main__":
    main()
