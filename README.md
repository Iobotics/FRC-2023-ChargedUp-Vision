# FRC-2023-ChargedUp-Vision

Pi should have onboard LED at red if power is going in
On startup there should be a green LED
if green LED not showing up SD card or power is incorectly plugged in

Program should run automatically on startup
If camera feed is not sending to dashboard kill background service in the vision tab 
startup after kill program takes around 10 seconds

On startup pi will ask for user login.
RaspberryPi User Login:
    Username: pi
    Password: raspberry

Ethernet should plug into the radio
Connect to radio wifi
Open http://wpilibpi.local/ in web browser for interface and uploading files

Go to Application and go to File Upload to upload new files

RaspberryPi should have file cameraServer.py
python3 cameraServer.py to start sending webcam data
sends data to 10.24.38.2

Make sure that shuffleboard is set to correct networktables ip
File -> Preferences -> NetworkTables and set server to 10.24.38.3
If widgets are not showing up automatically the camera feed can be found on the left after clicking the 2 arrows
Under CameraServer device USB Camera 0 should show up

if Camera is not connecting run sudo svc -t /service/camera to reset camera

