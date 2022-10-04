'''
Credits to: Sergia Canu
Source: https://pysource.com/2021/05/28/measure-size-of-an-object-with-opencv-aruco-marker-and-python/
'''
#%%
import numpy as np
import cv2,time,glob,os
from cameraLogitech import CameraSetup
from xarm.wrapper import XArmAPI

#Create aruco object instance
parameters = cv2.aruco.DetectorParameters_create()
parameters.adaptiveThreshConstant = 5
parameters.useAruco3Detection = True
parameters.polygonalApproxAccuracyRate = 0.0042
parameters.aprilTagMinClusterPixels = 50
parameters.aprilTagMaxNmaxima = 5
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
# Get Aruco marker
img = cv2.imread(r"C:\Users\kong.kah.chun\Documents\KKC_Documents\xArm Project\xArm Lite 6\xArmPythonSDK\xArm-Python-SDK\aruco_test\test_5.jpg")

# Undistort image
cam = CameraSetup()
dst = cam.undistortAndCrop(img)

#Obtain marker corners
_, _, corners = cv2.aruco.detectMarkers(dst, aruco_dict, parameters=parameters)

# Draw polygon around the marker
int_corners = np.int0(corners)
cv2.polylines(dst, int_corners, True, (0, 255, 0), 5)

#Find the center of marker
marker_center = (np.sum(corners[0],axis=1)/4).astype(int)
marker_center = tuple(map(tuple,marker_center))
cv2.circle(dst,marker_center[0],radius=5,color=(0,255,0),thickness=-1)

#Find the image center
image_center = (np.flip(np.array(dst.shape[:2])/2)).astype(int)
cv2.circle(dst,image_center,radius=5,color=(0,0,255),thickness=-1)

#Calculate xy distance from image center
xy_distance = marker_center - image_center

#Draw a line in between the centers
cv2.line(dst,marker_center[0],image_center,(255,0,0),2)

# Aruco Perimeter
aruco_perimeter = cv2.arcLength(corners[0], True)

# Calculate mm to pixel ratio
mm2pixel = 200 / aruco_perimeter

#Calculate distance in mm to move to marker center
xy_distance_mm = xy_distance * mm2pixel

#Compute Euclidean distance (displacement)
displacement = np.linalg.norm(xy_distance_mm,axis=1)

#%%
#Show image
img = cv2.resize(img,(1080,720))
dst = cv2.resize(dst,(1080,720))
cv2.imshow("aruco original",img)
cv2.waitKey(0)
cv2.imshow("aruco undistorted",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
#Initialize robot
robot_ip = '192.168.1.224'
response = os.popen(f'ping {robot_ip}').read()
assert not (('Request timed out.' or 'unreachable') in response), 'Target IP unreachable'
print(response)

arm = XArmAPI(robot_ip,is_radian=False)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)
speed = 200
mvacc = 2000

#Obtain current robot position
current_position = np.array(arm.position) 

#Compute distance to move for robot arm
move_distance = (np.flip(xy_distance_mm)).flatten() * -1
#new_position = np.add(current_position,move_distance)

#%%
#Move robot to new position
arm.set_position(x=move_distance[0],y=move_distance[1],speed=speed,mvacc=mvacc,relative=True,wait=True)
arm.disconnect()
# %%
webcam = cv2.VideoCapture(3)

#Check if robot is connected
robot_ip = '192.168.1.224'
response = os.popen(f'ping {robot_ip}').read()
assert not (('Request timed out.' or 'unreachable') in response), 'Target IP unreachable'
print(response)

arm = XArmAPI(robot_ip,is_radian=False)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)
speed = 200
mvacc = 2000
arm.set_position(*[250.0, 0.0, 450.0, 180.0, 0.0, -90.0], speed=speed, mvacc=mvacc, radius=0.0, wait=True)

while webcam.isOpened():
    ret, frame = webcam.read()
    if not ret:
        break
    
    #Convert the frame into numpy array
    image_np = np.array(frame)
    dst = cam.undistortAndCrop(image_np)
    #Obtain marker corners
    _, _, corners = cv2.aruco.detectMarkers(dst, aruco_dict, parameters=parameters)

    if not corners:
        pass
    else:
        # Draw polygon around the marker
        int_corners = np.int0(corners)
        cv2.polylines(dst, int_corners, True, (0, 255, 0), 5)

        #Find the center of marker
        marker_center = (np.sum(corners[0],axis=1)/4).astype(int)
        marker_center = tuple(map(tuple,marker_center))
        cv2.circle(dst,marker_center[0],radius=5,color=(0,255,0),thickness=-1)

        #Find the image center
        image_center = (np.flip(np.array(dst.shape[:2])/2)).astype(int)
        cv2.circle(dst,image_center,radius=5,color=(0,0,255),thickness=-1)

        #Draw a line in between the centers
        cv2.line(dst,marker_center[0],image_center,(255,0,0),2)

        #Calculate xy distance from image center
        xy_distance = marker_center - image_center

        # Aruco Perimeter
        aruco_perimeter = cv2.arcLength(corners[0], True)

        # Calculate mm to pixel ratio
        mm2pixel = 200 / aruco_perimeter

        #Calculate distance in mm to move to marker center
        xy_distance_mm = xy_distance * mm2pixel

        #Compute Euclidean distance (displacement)
        displacement = np.linalg.norm(xy_distance_mm,axis=1)
        displacement = np.round(displacement,decimals=2)

        #Put text onto the image
        cv2.putText(dst,f'{displacement[0]}',(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        if arm.state == 2:
            if displacement[0] > 5:
                #Compute distance to move for robot arm
                move_distance = (np.flip(xy_distance_mm)).flatten() * -1
                #Move robot to new position
                arm.set_position(x=arm.position[0]+move_distance[0],y=arm.position[1]+move_distance[1],speed=speed,mvacc=mvacc,relative=False,wait=False)
                # time.sleep(0.3)
            

    #Show image
    cv2.imshow('Webcam undistorted test',dst)
    
    if cv2.waitKey(10) & 0XFF == ord('q'):
        break

cv2.destroyAllWindows()    
webcam.release()
arm.disconnect()
# %%
