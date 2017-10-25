import numpy as np
import cv2
from matplotlib import pyplot as plt


""" watershed: performs the watershed algorithm on the given image. 
      Code is taken from an OpenCV tutorial on the Watershed algorithm which 
      can be found here: http://docs.opencv.org/3.2.0/d3/db4/tutorial_py_watershed.html
"""
#TODO: expand on the comment and cite this
def watershed(img, gray, ret, thresh):
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    # look at the region of the central pixel 
    rows, cols = markers.shape
    center_group = markers[rows/2, cols/2]
    mask = np.zeros(markers.shape, dtype="uint8")
    mask[np.where(markers==center_group)] = 255 
    #cv2.imwrite('watershed.jpg', mask)
    return mask


""" watershed_img: helper function to run watershed in either RGB or HSV space
    params: 
      img - numpy array containing a 3-channel image assumed to be in RGB space
      hsv_space - boolean indicating if the image should be converted to HSV space
        if False, the image will remain in RGB space
    returns:
      a numpy array containing a binary mask / segmentation
""" 
def watershed_seg(img, hsv_space=False):
    if hsv_space:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        #NOTE: maybe 0 - 255 should be 127 to 255?  also, changing the types of thresholds changes things
        ret, thresh = cv2.threshold(gray[:,:,0],0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    else:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return watershed(img.copy(), gray, ret, thresh)


