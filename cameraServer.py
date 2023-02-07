import cscore
import ntcore
import cv2
import numpy as np
import time
import april_detector

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

# def processing(img): #returns and accepts bgr image
#     hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     cube_img = cv2.inRange(hsvimg, (120,100,100),(160,255,255))
#     cts = contours(cube_img)
#     output_img = draw(img,cts,(255,0,0))

#     hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     cone_img = cv2.inRange(hsvimg, (20,150,150),(40,255,255))
#     cts = contours(cone_img)
#     output_img = draw(output_img,cts,(255,0,0))

#     return output_img

def main():
    april = april_detector.AprilDetector()

    cameraSelected = 0
    cs = cscore.CameraServer
    cs.enableLogging()

    camera_1 = cscore.UsbCamera(name = 'camera_1', dev = 0)
    camera_1.setResolution(1280,960)
    camera_2 = cscore.UsbCamera(name = 'camera_2', dev = 2)
    camera_2.setResolution(1280,960)

    cs.addCamera(camera_1)
    cs.addCamera(camera_2)

    # Does not actually send video back to dashboard
    # Creates cvSource object that is used to send frames back
    outputSource = cs.putVideo('feed',1280,960)

    #creates cvSink object that can be used to grab frames
    outputSink1 = cs.getVideo(camera=camera_1)
    outputSink2 = cs.getVideo(camera=camera_2)

    
    #pre allocate img for memory
    img1 = np.zeros(shape=(1280, 960, 3), dtype=np.uint8)
    img2 = np.zeros(shape=(1280, 960, 3), dtype=np.uint8)

    #initialise network tables instance
    nt = ntcore.NetworkTableInstance.getDefault()
    nt.setServerTeam(2438)
    nt.startClient4("temp")

    sd = nt.getTable("SmartDashboard")
    
    translationX = sd.getDoubleTopic("TranslationX")
    translationY = sd.getDoubleTopic("TranslationY")
    translationZ = sd.getDoubleTopic("TranslationZ")
    rotationX = sd.getDoubleTopic("RotationX")
    rotationY = sd.getDoubleTopic("RotationY")
    rotationZ = sd.getDoubleTopic("RotationZ")
    
    translationXPublisher = translationX.publish()
    translationYPublisher = translationY.publish()
    translationZPublisher = translationZ.publish()
    rotationXPublisher = rotationX.publish()
    rotationYPublisher = rotationY.publish()
    rotationZPublisher = rotationZ.publish()
    while True:
        # Grabs latest frame and sets to img, out returns 0 if error and time if not error
        out, img1 = outputSink1.grabFrame(img1)

        if out == 0:

            print(outputSink1.getError())
            
            continue
        
        out, img1 = april.detect(img1)

        if out != 0:
            print(out)
            translationXPublisher.set(value = out[0])
            translationYPublisher.set(value = out[1])
            translationZPublisher.set(value = out[2])
            rotationXPublisher.set(value = out[3])
            rotationYPublisher.set(value = out[4])
            rotationZPublisher.set(value = out[5])
        
        #out, img2 = outputSink2.grabFrame(img2)

        #if out == 0:
        #    print(outputSink2.getError())
            
        #    continue
        
        #img2 = aprtags(img2)
        
        img1 = cv2.resize(img1, (360,240))
        outputSource.putFrame(img1)




if __name__ == "__main__":

    main()
