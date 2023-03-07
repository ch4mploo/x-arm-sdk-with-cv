"""
xArm SDK Test with OpenCV

This is a program to test the Python SDK of xArm using computer vision application.
Full credit of xArm Python SDK goes to : https://github.com/xArm-Developer/xArm-Python-SDK

"""
import numpy as np
import cv2,time,glob,os,gc
from cameraLogitech import CameraSetup
from xarm.wrapper import XArmAPI
import streamlit as st

st.title("Aruco Marker Tracker")
st.header("1. Fine tune Aruco marker detector parameters")
arucoAtt = np.array(dir(cv2.aruco))
arucoType = arucoAtt[['DICT_' in x for x in arucoAtt]]
with st.form('aruco_form'):
    camera_port = st.number_input('Enter camera port number',min_value=0,max_value=5,value=3,step=1)
    adaptiveThreshold = st.slider("Adaptive Threshold Constant",min_value=1,max_value=10,step=1,value=7)
    polygonAcc = st.slider("Polygonal Approximate Accuracy Rate",min_value=0.001,max_value=0.050,step=0.001,format='%3f',value=0.005)
    minClustPixel = st.slider("Minimal Pixel Cluster Rejection",min_value=5,max_value=10,step=1)
    maxNmaxima = st.slider("Maximum number of corners for detected polygons",min_value=4,max_value=10,step=1)
    arucoTypeSelection = st.selectbox("Type of Aruco marker",arucoType.tolist())
    submit_aruco = st.form_submit_button('Update Aruco Parameters')

st.header('2. Tune xArm parameters')

with st.form('xarm_form'):
    speed = st.slider('Robot speed',min_value=10,max_value=255,step=5,value=200)
    mvacc = st.slider('Robot Acceleration',min_value=100,max_value=2000,step=50,value=2000)
    robot_ip = st.text_input('xArm IP address',value='192.168.1.224',placeholder='IPv4 address')
    submit_xarm = st.form_submit_button('Update xArm Parameters')
    if submit_xarm:
        #Check if robot is connected
        response = os.popen(f'ping {robot_ip}').read()
        assert not (('Request timed out.' or 'unreachable') in response), 'Target IP unreachable'
        st.write('Robot connected')
#Aruco parameters
parameters = cv2.aruco.DetectorParameters_create()
parameters.adaptiveThreshConstant = adaptiveThreshold
parameters.useAruco3Detection = True
parameters.polygonalApproxAccuracyRate = polygonAcc
parameters.aprilTagMinClusterPixels = minClustPixel
parameters.aprilTagMaxNmaxima = maxNmaxima
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.__getattribute__(arucoTypeSelection))

#Setup camera parameters

cam = CameraSetup()
webcam = cv2.VideoCapture(camera_port)
if not (webcam.isOpened()):
    st.error('Webcam not detected')
    st.stop()

@st.cache_data      #Add streamlit data cache
def draw_and_move(camera,x_arm, webcam_frame):
    #Convert the frame into numpy array
    image_np = np.array(webcam_frame)
    dst = camera.undistortAndCrop(image_np)
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
                x_arm.set_position(x=x_arm.position[0]+move_distance[0],y=x_arm.position[1]+move_distance[1],speed=speed,mvacc=mvacc,relative=False,wait=False)
                # time.sleep(0.3)        
    return dst

@st.cache_data      #Add streamlit data cache
def start_webcam_stream(image_frame):
    stop_button = st.button("Stop Process")
    while webcam.isOpened():
        ret, frame = webcam.read()
        if not ret:
            break
        
        dst = draw_and_move(cam,arm,frame)
        
        #Show image
        dst = cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)
        image_frame.image(dst,clamp=True)
        dst, frame = None, None     #clear dst and frame np
        del dst, frame  #delete the variables
        if stop_button:
            webcam.release()
            arm.disconnect()
            gc.collect()    #garbage collector to free unallocated space
            break

st.header('3. Start the process')
placeholder_img = cv2.imread(r".\aruco_test\no_aruco.jpg")
start_button = st.button("Connect xArm and start camera")
if start_button:
    arm = XArmAPI(robot_ip,is_radian=False)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
    arm.set_position(*[250.0, 0.0, 450.0, 180.0, 0.0, -90.0], speed=speed, mvacc=mvacc, radius=0.0, wait=True)

    image_frame = st.image(placeholder_img,channels='BGR',clamp=True)

    start_webcam_stream(image_frame)

# else:
    # webcam.release()
    # arm.disconnect()

#%%

