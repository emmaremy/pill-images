from sys import *
import numpy as np
from sklearn import cluster
import collections
import cv2

def get_img_hist(img):
  print "clustering..."
  img = np.array(img)
  pixels = np.reshape(img, (-1, 3))
  kmeans = cluster.KMeans(n_clusters=3, precompute_distances=True)
  kmeans.fit(pixels)
  print "finished clustering!"

  labels = kmeans.labels_
  centroids = kmeans.cluster_centers_

  # get a count of the number of elements associated with each label
  cluster_sizes = collections.Counter(labels)

  # make centroid, count pairs 
  feature_vectors = []
  print "creating feature vectors"
  for i in cluster_sizes:
    feature_vec = list(centroids)
    feature_vec.append(cluster_sizes[i])
    feature_vectors.append(feature_vec)

  new_img = visualize_colors(labels, centroids, img.shape)

  return feature_vectors, new_img


def visualize_colors(labels, colors, shape):
  print "creating the new image"
  new_image = []
  for label in labels:
    new_image.append(colors[label])
  new_image = np.array(new_image).reshape(shape)
  return new_image

