# Testing xArm Python SDK with Computer Vision Technology

## 1. Overview
xArm is a collaborative robot made by Ufactory. The Python SDK for this robot allows users to freely program the movement and behaviour of the robot. This project also uses Streamlit to create an interactive application for users to control the parameters of the robot and the computer vision algorithm.

For detailed documentation about xArm Python SDK, please visit their GitHub repository:
https://github.com/xArm-Developer/xArm-Python-SDK

## 2. Dependencies requirement
This project uses Python version 3.8.13
```
pyserial==3.4
opencv-contrib-python
numpy
streamlit
```

## 3. Installation and running the application
(A) No installation needed, just git clone this project:
```
git clone https://github.com/ch4mploo/x-arm-sdk-with-cv.git
```
(B) Once the project is cloned, you can run the Streamlit app with this command:
```
streamlit run app.py
```
Note: Make sure you have the dependencies needed. It is recommended to create your own Python virtual envinronment such as using Anaconda.Before you run the command, also make sure you have changed your directory into where the project is located.

(C) When the app starts to run, a webpage should automatically appear. If it doesn't, you can open the app manually by typing **localhost:8501** in your web browser. If the webpage for the app still doesn't appear, please check your console logs for errors.

(D) Enjoy using the app!
