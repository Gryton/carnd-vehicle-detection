** Vehicle Detection Project - Mateusz Gryt's Submission**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

Here it is.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I explored different color spaces, different spatial sizes for spatial histograms, different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=8` and `cells_per_block=2`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters

I ended with `orientations=9`, `pixels_per_cell=8` and `cells_per_block=2`, as I found that greater values made worse
accuracy on test set. Using such parameters I generally had over 99% accuracy on test set (if I haven't done other
parameters for colors totally wrong), so I thought it might be good base for going further.

I tried various combinations of parameters and

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using features from all HOG channels, color histogram features and spatial color binning. Last one
looked pretty promiscuous, as I wanted to use HLS color space. After extracting all features I normalized the data using
`sklearn.preprocessing.StandardScaler`.

It gave me nice accuracy on test dataset, as I achieved 99.5% of accuracy.

After all I've dumped my trained SVC and training parameters to pickle file, to get some module encapsulation, and feel
more comfortable with testing different possibilities, as exactly the same SVC if I mixed the searching pipeline, or I
could change SVC and run on pipeline providing directly the same parameters.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I saw, that road is generally between 400 and 656 y pixels, so I masked this region for looking for cars within. Then I tried
different windowing, starting from 16x16 and ending with 128x128. It seemed that window 16x16 generally has no sense,
and it takes very long time for computing, as there will be many windows to compute. Windows 64 and 96 seemed to give good
output, so I thought that I can use it in my pipeline. When I changed algorithm to compute HOG only once per frame (not
per window) and instead of resizing window resizing image, I ended with scales 1, 1.5, 2, 2.5, 3, what effectively means
windows 64, 96, 128, 160, 192, as larger windows seem to have low rate of false positives, and are fast to compute, and
improve a bit found cars (better fitting bounding boxes). I thought that overlaping windows by half should give me good
results, as no objects should be missed. By now, I think that higher overlap might be beneficial, but it would add extra
time for computing - even without it i took me over 1 second for frame, so it wouldn't be welcomed.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

It was very confusing, when I ended with really not good result, when having 99.3% accuracy on test set. Below is example
of image, of using my trained classifier.
![alt text][image4]

After playing with different parameters I saw that using HLS color space wasn't good idea at all. I ended with YCrCb
space, and then my result for the same image was much better:
![alt text][image4]

Ultimately, I used scales [1, 1.5, 2, 2.5, 3] plus color histogram plus spatially binned color.


![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.
From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.
I've implemented a class, that will memorize n previous heatmaps and average them, to have more smooth
image, get rid of single frame false positives, and "find" cars throughout whole series where images are lost for single
frame.
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.
I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

