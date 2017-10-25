# Small code to transform our data to something functional

from sys import *
from collections import defaultdict
""" get_other_ref: given the image file name of one reference image in the data
      set, get the file name of the other reference image. 
    params:
      one_ref - a string containing the filename or path of one of the images
    returns:
      a string containing the filename or path of the other reference image
"""
def get_other_ref(one_ref):
  if "_SB_" in one_ref:
    return one_ref.replace("_SB_", "_SF_")
  elif "_SF_" in one_ref:
    return one_ref.replace("_SF_", "_SB_")
  else:
    print >> sys.stderr, "invalid reference image name"


""" get_all_refs: returns a list containing the filenames for all of the 
        reference images in the dataset.
      params:
        d - a dictionary mapping one reference image to all of the customer
          quality images associated with that reference image
      returns - a list of strings containing the filenames for all of the 
        reference images. 
"""
def get_all_refs(d):
  orig_refs = d.keys()
  refs = []
  for k in orig_refs:
    refs.append(k)
    refs.append(get_other_ref(k))
  return refs


""" get_all_custs: returns a list containing the filenames for all of the 
        customer quality images in the dataset.
      params:
        d - a dictionary mapping one reference image to all of the customer
          quality images associated with that reference image
      returns - a list of strings containing the filenames for all of the 
        customer quality images. 
"""
def get_all_custs(d):
  orig_custs = d.values()
  return [item for sublist in orig_custs for item in sublist]


""" data_transform: transforms the data read in from the ground truth table 
        file to a dictionary mapping the name of one references image to a list
        of the names of all of the customer quality images associated with that
        reference images. 
      params:
        filename - string containing the path to the file in which the ground
          truth table is stored
      returns:
        a dictionary from string to string list as described above. 
"""
def data_transform(filename):
  try:
    f = open(filename, 'r')
  except IOError:
    print "Could not open file \"%s\".  Exiting." % filename
    exit()
  lines = f.readlines()
  data = defaultdict(list)
  for i in range(1, len(lines)):
    lines[i] = lines[i].split("\",\"")
    data[lines[i][2]].append(lines[i][1])
  return data
