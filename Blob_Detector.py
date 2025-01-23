import cv2
import numpy as np
import os

def init_params():
    params = cv2.SimpleBlobDetector_Params()
    return params

def init(params=None):
    if params is None:
        # Default parameters
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 2000
        params.filterByCircularity = False
        params.filterByConvexity = True
        params.minConvexity = 0.87
        params.filterByInertia = True
        params.minInertiaRatio = 0.01
    
    # Create a detector with the given parameters
    detector = cv2.SimpleBlobDetector_create(params)
    return detector
    
def detect(detector, img):
    keypoints = detector.detect(img)
    return keypoints

def show_blob(keypoints, img):
    return cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)