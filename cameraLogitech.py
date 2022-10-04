import cv2,os
import numpy as np

class CameraSetup:
    def __init__(self):
        self.save_path = r'.\camera_properties\c922'
        self.cameraMatrix = np.load(os.path.join(self.save_path,'cameraMatrix.npy'))
        self.dist = np.load(os.path.join(self.save_path,'dist.npy'))

    def undistortAndCrop(self,img):
        h,w = img.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(self.cameraMatrix, self.dist, (w,h), 1, (w,h))
        dst = cv2.undistort(img, self.cameraMatrix, self.dist, None, newCameraMatrix)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst


