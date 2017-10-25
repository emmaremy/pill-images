"""
  main program for our pill classifier

  Alison Rosenzweig
  Emma Remy
"""

from sys import *
import glob # used to figure out which images were hand segmented
import numpy as np
import cv2
import image_segmentation as imgs
import watershed as ws
import data_transform as dt
import clustering as cl
import evaluation 
import csv

# params: ground truth filename, image file

def main():

  if len(argv) != 2:
    print "Incorrect number of arguments."
    print "Usage: python pill_segmentation.py <ground truth table>"
    exit()
  # get dictionary of ref name -> all customer image names
  data_file = argv[1]
  data = dt.data_transform(data_file)
  refs = dt.get_all_refs(data)


  ref_src_directory = "/data/cs68/pillchallenge/dr/"
  cust_src_directory = "/data/cs68/pillchallenge/dc/"

  # example applying a segmentation technique to all of the reference images
  # Note that this takes several minutes to run. 
  """
  segment_all_refs(refs, 
      imgs.distance_to_median_seg, 
      "/scratch/arosenz1/cs68_final_proj/distance_from_median/", 
      ref_src_directory)
  """
  
  hand_seg_directory = "/scratch/arosenz1/cs68_final_proj/hand_segmented/"
  hand_seg_list = get_hand_seg_list(hand_seg_directory)

  custs = dt.get_all_custs(data)

  # example calculating the Sorensen-Dice Index metrics for customer-quality
  # images 
  calc_metrics(
      custs, ws.watershed_seg, 'output/cust_watershed_hsv.csv', 
      hand_seg_list, cust_src_directory, hand_seg_directory, hsv_space=True) 
  

  # example creating comparison figures for reference images using 
  # the distance to average segmentation. 
  for img in refs:
    if img in hand_seg_list:
      name, extension = img.split(".")
      make_comparison_figure(
          img, cust_src_directory, hand_seg_directory, 
          imgs.distance_to_avg_seg, hand_seg_list, 
          name + "_dist_to_avg_figure." + extension, True, True)


""" scale_down_img: scales an image down to the given height """
def scale_down_img(img, end_height):
  width, height = img.shape[:2]
  ratio = float(end_height)/height
  return cv2.resize(img, (int(height*ratio), int(width*ratio)), interpolation=cv2.INTER_CUBIC)


""" make_comparison_figure: creates an image with the programatically segmented
      image on the left, the original image in the middle, and the 
      corresponding manually segmented image on the right
"""
def make_comparison_figure(
    image_file_name, source_directory, hand_seg_directory, segmentation_function, hand_seg_list,
    output_file_name, include_orig=False, hsv_space=False):
  
  print "Creating a figure for %s to be saved in file %s" % (image_file_name, output_file_name)
  orig_img = cv2.imread(source_directory + image_file_name)
  name, extension = image_file_name.split(".") 
  prog_seg = segmentation_function(orig_img, hsv_space)
  
  # scale the images to all be the same height
  scale = 300
  orig_img = scale_down_img(orig_img, scale)
  prog_seg = scale_down_img(prog_seg, scale)
  print prog_seg.shape
  
  height, width, channels = orig_img.shape
  buffer_shape = (height, 10, channels)
  buffer_img = 255*np.ones(buffer_shape)

  # convert the segmented image into a 3-channel rgb image
  prog_seg = cv2.cvtColor(prog_seg, cv2.COLOR_GRAY2BGR)
  
  include_hand_seg = True
  if image_file_name not in hand_seg_list:
    include_hand_seg = False
    print "This image doesn't have a hand-segmented counterpart. \n " \
        "Will create a figure with only the original and the segmented images"
    hand_seg = np.array([])
  else:
    hand_seg = cv2.imread(hand_seg_directory + name + "_seg." + extension, cv2.IMREAD_GRAYSCALE)
    # convert the grayscale hand segmented image to a binary b&w image
    thresh, hand_seg = cv2.threshold(hand_seg, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    hand_seg = scale_down_img(hand_seg, scale)
    hand_seg = cv2.cvtColor(hand_seg, cv2.COLOR_GRAY2BGR)

  if include_orig:
    if include_hand_seg:
      compound_img = np.concatenate((prog_seg, buffer_img, orig_img, buffer_img, hand_seg), axis=1)
    else:
      compound_img = np.concatenate((prog_seg, buffer_img, orig_img), axis=1)
  elif image_file_name in hand_seg_list:
    compound_img = np.concatenate((prog_seg, buffer_img, hand_seg), axis=1)
  else:
    print "Figure would only contain programatic segmentation.  No figure being created."
    return

  cv2.imwrite(output_file_name, compound_img)


""" segment_all_refs: segments the reference images (from the list of file 
      names passed) using the passed segmentation function and stores the
      results in the destination directory. 
    params:
      refs - list of image file names
      segmentation_function - function that takes a 3 channel image and returns
        a binary mask
      dest_directory - string indicating the directory in which results should be stored
      source_directory - directory in which the input images are stored
"""
def segment_all_refs(refs, segmentation_function, dest_directory, source_directory):
  # segment all of the reference images and store them in scratch
  # list of reference images 
  for image_name in refs:
    img = cv2.imread(source_directory + image_name)
    segmented = segmentation_function(img)
    cv2.imwrite(dest_directory + image_name, segmented)


""" calc_metrics: given a list of image file names and a source directory, 
      calculates Dice's index for any of the images for which a hand segmented 
      image exists using the given segmentation function. 
    params:
      refs - list of image file names
      segmentation_function - function that takes a 3 channel image and returns
        a binary mask
      output_data_file - string containing the file name and path where output
        data in CSV format should be stored
        CSV rows will contain an image file name and the corresponding Dice Index. 
      hand_seg_list - list of all the image file names of images that have a 
        corresponding manually segmented image
      source_directory - directory in which the input images are stored
      hand_seg_directory - directory in which the hand segmented images are stored
""" 
def calc_metrics(refs, segmentation_function, output_data_file, hand_seg_list, 
    source_directory, hand_seg_directory, hsv_space=False):
  # make a dictionary mapping the image file name to the result of the Dice's coefficient calculation 
  dices_data = {} 
  for image_name in refs:
    if image_name in hand_seg_list:
      img = cv2.imread(source_directory + image_name)
      name, extension = image_name.split(".") 
      prog_seg = segmentation_function(img, hsv_space)
      hand_seg = cv2.imread(hand_seg_directory + name + "_seg." + extension, cv2.IMREAD_GRAYSCALE)
      # convert the grayscale hand segmented image to a binary b&w image
      thresh, hand_seg = cv2.threshold(hand_seg, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
      result = evaluation.dices(prog_seg, hand_seg)
      dices_data[image_name] = result
  with open(output_data_file, 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in dices_data.items():
      writer.writerow([key, value])
  print dices_data
  

""" get_hand_seg_list: returns a list of all of the image files (by name)
      for which corresponding manually segmented images exist.
    params: 
      directory - string indicating the directory in which the hand segmented
        images are stored
    returns:
      a list of strings containing image file names as described above
"""
def get_hand_seg_list(directory):
  seg_files_long = glob.glob(directory + "*.jpg")
  seg_files = [long_name.split("/")[-1] for long_name in seg_files_long]
  # cutout "_seg" at the end of the file names
  seg_files = [name[0:-8] + ".jpg" for name in seg_files]
  return seg_files


if __name__ == "__main__":
  main()
