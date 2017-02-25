

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/cars_non_cars.png
[image2]: ./output_images/car_ch_0.png
[image3]: ./output_images/car_ch_1.png
[image4]: ./output_images/car_ch_2.png
[image5]: ./output_images/not_car_0.png
[image6]: ./output_images/Not_car_ch_1.png
[image7]: ./output_images/Not_car_ch_2.png
[image8]: ./output_images/detect_car.png
[image9]: ./output_images/detect_no_car.png
[image10]: ./output_images/multiple_image_heatmap.png
[image11]: ./output_images/image_boundary_label.png
[image12]: ./output_images/detection_heatmap.png
[video1]: ./final_video.mp4





[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook (`vehicle-detection.ipynb`).  

Read all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Explored different color spaced and used `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=8` and `cells_per_block=2`. Used CV2's hog method to extract HOG features for each channel. Following images represents the images from each channel and its corresponding HOG representation.


![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of color space from RGB, HLS, LUV, but finally settled on `YCrCb`as it provided the best results when used with the following

orient = 9  # HOG orientations

pix_per_cell = 8 # HOG pixels per cell

cell_per_block = 2 # HOG cells per block

hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

spatial_size = (32,32) # Spatial binning dimensions

hist_bins = 32    # Number of histogram bins

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Extracted features using spatial binning, color histogram  and hog. Created a feature vector for every image and used Linear SVM to classify as cars vs not cars.

This can be seen in 3rd code cell of the ipython notebook

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First reduced the area to investigate lower by checking y between 400 to 656, Since we don't need to check at the top of the picture
Then tried various scales from 1 to 2. Settled on 1.5 based on the performance
Used step of 2 which will be given 75% overlap with window size of 64. This was able to give maximum performance



####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Initially, i was getting hog features for every sliding window, but this proved to be very slow. Each frame was taking about 20 seconds. Then extracted HOG features for the entire image and performed sliding window on that. This resulted in huge performance optimization, per second, it was doing about 2 frames.

Following are examples of pipeline images

###Detected Cars
![alt text][image8]

###No cars found
![alt text][image9]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./final_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

By just using sliding window technique for detecting a car, there are chances of False positives. Inorder to remove that, i have used heatmap technique to draw a heatmap and applying threshold of 1 so that only when multiple detections are made, that particular window will be selected. For finding the boundary of overlapping window, i used 'scipy.ndimage.measurements.label()' to label the image. Then, i got the bounding window and applied to the image. This can be found in 8th code cell.

### Image and Heatmap

![alt text][image10]

### Here is the output of `scipy.ndimage.measurements.label()`
![alt text][image11]

### Here the resulting bounding boxes after eliminating false positives and overlapping boundaries
![alt text][image12]

## Checking multiple frames

I also implemented a way to check two frames to detect a car so that false positives can be eliminated. This can be found in 6th code cell
---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Initially, i was getting hog features for every sliding window, this proved to be too costly and the resultant perforance was very poor. It took 20 seconds for each frame though its accuracy was good. Then i changed this to extract HOG only once for the image and extracted subarray from it for each sliding window. This helped a lot in performance. Also, i spent some time chosing color spaces and other parameters needed for HOG.

when the car on the right is little farther, pipeline is unable to detect. Need to spend some time to be able to detect that.
