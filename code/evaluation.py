import numpy as np

""" dices: calculates Sorensen-Dice index / Dice's coefficient / the Sorensen 
      index for two binary images / masks
    params:
      seg1 - numpy array containing the first binary image to be compared
      seg2 - numpy array containing the second binary image to be compared
    returns:
      a float in [0, 1] representing the Sorensen-Dice index
"""
def dices(seg1, seg2):
  overlap = np.logical_and(seg1, seg2)
  overlap_size = np.count_nonzero(overlap)

  seg1_size = np.count_nonzero(seg1)
  seg2_size = np.count_nonzero(seg2)
  print "overlap size: %d, seg 1 size: %d, seg 2 size: %d" % (overlap_size, seg1_size, seg2_size) 
  return 2*float(overlap_size)/(seg1_size + seg2_size)
