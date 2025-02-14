# Project Title
Usage:

segmented_images/skin_contours/coords
-- contains coordinates of detected facial skin contour, including face and nose.

pipeline.py
-- process image with various color channels and create masks based on distribution.
-- YCRCB, LAB, SAT_BLUE: color channel for selection
-- FILTER: create mask based on various channels for contour detection
-- SAVE_MASK: save contour detection image
-- DISPLAY: show all processed image

weight_X_list: list of weights of color channel X
weight_X: numerical weights of color channel X, need to match the number in list
min_area: minimal number of pixels in the contour

process_X.py
-- process image in color channel X, check distribution, create mask

contours_group.py
-- group detected contours in user-defined bounding box, currently 1024x1024
