import numpy as np
import cv2
import scipy.spatial.distance as sp_dist

""" threshold_seg: creates image segmentations using simple thresholding 
      techniques along with morphological operations.
    params:
      img - a numpy array containing a 3-channel image
      hsv_space - a boolean indicating if the image should be converted to hsv
        space.  If False, the image will instead be converted to grayscale.
    returns:
      a numpy array containing the segmentation / binary mask
"""
def threshold_seg(img, hsv_space=True):
  #img = cv2.imread(img_file_name)
  print img
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  if hsv_space:
    ret, mask = cv2.threshold(hsv[:,:,0], 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  else:
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  smoothed = morph_ops(mask)
  #_, contours, hierarchy = cv2.findContours(smoothed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  #smoothed = cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)
  
  #cv2.drawContours(smoothed, contours, -1, (0, 255, 0), 5)
  # cv2.imwrite('contours.jpg', smoothed)
  return smoothed


""" morph_ops: performs a series of morphological operations on the passed
      mask to reduce noise and smooth out the boundary lines.
    params:
      mask - numpy array containing a binary mask 
    returns: 
      a numpy array containing a (hopefully smoothed and de-noised) binary mask
"""
def morph_ops(mask):
  little_kernel = np.ones((3, 3), np.uint8)
  kernel = np.ones((5, 5), np.uint8)
  big_kernel = np.ones((9, 9), np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, little_kernel, iterations=1)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, big_kernel, iterations=5)
  smoothed = cv2.blur(mask, (19, 19))
  ret, smoothed = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  return smoothed


""" distance_to_avg_seg: creates image segmentations by thresholding on the 
      distance from the average pixel value in the image along with 
      morphological operations to remove noise and smooth out the mask
    params:
      img - a numpy array containing a 3-channel image
      hsv_space - a boolean indicating if the image should be converted to hsv
        space.  If False, the image will remain in RGB space.
    returns:
      a numpy array containing the segmentation / binary mask
"""
def distance_to_avg_seg(img, hsv_space=False):
  if hsv_space:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  col_avgs = np.average(img, axis=1)
  avg_color = np.average(col_avgs, axis=0)
  img_minus_avg = np.abs(img - avg_color)
  img_minus_avg = np.array(img_minus_avg, dtype=np.uint8)
  dists = cv2.cvtColor(img_minus_avg, cv2.COLOR_BGR2GRAY)
  # Took out +cv2.THRESH_OTSU
  dret, thresh = cv2.threshold(dists, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  thresh = morph_ops(thresh)
  return thresh

""" distance_to_median_seg: creates image segmentations by thresholding on the 
      distance from the median pixel value in the image along with 
      morphological operations to remove noise and smooth out the mask
    params:
      img - a numpy array containing a 3-channel (RGB) image
      hsv_space - a boolean indicating if the image should be converted to hsv
        space.  If False, the image will remain in RGB space.
    returns:
      a numpy array containing the segmentation / binary mask
"""
def distance_to_median_seg(img, hsv_space=False):
  if hsv_space:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  pixels = np.reshape(img, (-1, 3))
  avg_color = np.median(pixels, axis=0)
  img_minus_avg = np.abs(img - avg_color)
  #img_minus_avg /= np.linalg.norm(img_minus_avg)
  #img_minus_avg *= 255
  img_minus_avg = np.array(img_minus_avg, dtype=np.uint8)
  dists = cv2.cvtColor(img_minus_avg, cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(dists, 0, 255, cv2.THRESH_BINARY)
  thresh = morph_ops(thresh)
  return thresh
  #cv2.imwrite('median.jpg', thresh)

"""
  new_img, contours = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

  print 'found', len(contours), 'contours'

  white = (255, 255, 255)

  display = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
  for j in range(len(contours)):
    cv2.drawContours(display, contours, j, white, -1)
 """
