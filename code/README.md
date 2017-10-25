### pill_segmentation.py

This file contains our main program as well as helper functions for generating csv files with output data and figures containing side-by-side results comparisons.  

The main program in this file is run with:

`python pill_segmentation.py <ground truth table>`

Where `<ground truth table>` is the path to a file containing the data mapping each customer image to both reference images for that pill. 

Our table is stored in the data directory, so the usage with our files and directory organization when run from the top level directory of this repository is:

`python code/pill_segmentation.py data/groundTruth_refs.csv`

The intent of this file (and our code generally) is to provide a library to easily use any of our segmentation metrics or our evaluation technique.  Our main program contains some examples of usage for the different functions (with some of the longer-running features commented out).  With this intention of library-style code, in addition to the example usages in `main`, our functions all have top level comments with descriptions of their functionality, parameters, and return values as documentation. 

### evaluation.py

This file contains the code for our evaluation method: the Sorensen-Dice Index.  It only contains one function which, given two segmentations of the same image, returns the S-D Index score for the pair. If we were to add more evaluation metrics in the future, they should also go in this library file (e.g. if we moved on to image classification rather than just segmentation). 

### image_segmentation.py

This file contains the bulk of our image segmentation functionality.  All of the functions in this file follow the same naming convention `<method>_seg` are called the same way:

`function_name(image, hsv_space)`

where the first argument is a numpy array containing a 3 channel image assumed to be in RGB space and the second is a boolean flag indicating if the image should be converted to HSV space before being evaluated.  

### watershed.py

Most of the code in watershed.py is from a OpenCV Tutorial (cited in the file) so, while it's an image segmentation method, it has its own file.  The code we modified is the end of the watershed function (adding code to create a binary mask from one of the regions watershed detects) and making the wrapper function conform to our conventions (calling it `watershed_seg` and accept a flag indicating if the analysis should be done in HSV space).

The Watershed segmentation should be executed by running the `watershed_seg` function from this library file with the same inputs as our other segmentation functions: an image as a numpy array and a boolean indicating the color space to be used. 

### data_transformation.py

This file contains the functions we used to transform the data provided that associated the images to eachother.  (Also note that some pre-processing was done in R). The functions that are most likely to be useful to other users of our code are `get_all_refs` and `get_all_custs`, which return lists of the filenames of all of the reference and customer-quality images respectively. 

### clustering.py

The code in this file is more related to our intentions for future work than it is to the rest of our code.  The `get_img_hist` function runs K-means (with k=3) on the pixels of the passed image and returns feature vectors with the 3 mean colors found in the image each with the size/number of pixels in associated cluster. The `visualize_colors` function wraps `get_img_hist` and replaces pixels with the mean of the cluster they're associated with as a way of visualizing the results of the clustering.  The `get_img_hist` and `visualize_colors` functions could be useful to future work related to learning distance metrics for classification or using k-means as another segmentation technique (respectively).  
