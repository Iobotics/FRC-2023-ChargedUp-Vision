import cscore
from networktables import NetworkTables
import logging
# Import OpenCV and NumPy
import cv2
import numpy as np

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


def main():
    cs = cscore.CameraServer.getInstance()
    cs.enableLogging()

    camera_1 = cscore.UsbCamera(name = 'camera_1', path = '/dev/video0')
    camera_1.setResolution(320,240)
    # camera_2 = cscore.UsbCamera(name = 'camera_2', path = '/dev/video2')

    cs.addCamera(camera_1)
    # cs.addCamera(camera_2)

    # Does not actually send video back to dashboard
    # Creates cvSource object that is used to send frames back
    outputSource = cs.putVideo('processed',320,240)

    #creates cvSink object that can be used to grab frames
    outputSink = cs.getVideo(camera=camera_1)

    #pre allocate img for memory
    img = np.zeros(shape=(1280, 960, 3), dtype=np.uint8)

    while True:
        # Grabs latest frame and sets to img, out returns 0 if error and time if not error
        out, img = outputSink.grabFrame(img)

        if out == 0:
            print(outputSink.getError())
            
            continue
        
        img = processing(img)

        outputSource.putFrame(img)


if __name__ == "__main__":
    main()