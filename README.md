# SKKU_ComputerVision

### prerequisite

numpy, opencv2

pip install numpy

## Project 1

It performs Gaussian Blur. Sigma value 1, 6, 11 and kernel size 5, 11, 17 are executed.

![1](https://user-images.githubusercontent.com/43103079/71712110-25cd7900-2e47-11ea-9ece-7c4b84a2f809.png)

Next, it performs Edge Detection for Gaussian Blurred image. 1D Edge detection kernel is used to detect horizontally and vertically.

![2](https://user-images.githubusercontent.com/43103079/71712984-089aa980-2e4b-11ea-8a16-c574f234579e.png)

Next, it performs Non-maximum-suppression. Each point verify for gradient direction which is descrted as 8 dirs. If gradient direction's image value is larger than center, the pixel is subject to be suppressed.

![3](https://user-images.githubusercontent.com/43103079/71712985-089aa980-2e4b-11ea-988d-b6719ec87c32.png)

Finally, it performs Corner detection by doing corner detection, non-maximum-suppression.




## Project 2

In CV_A2/A2_2d_transformation.py show an image and movement by transformation matrix. It is strong when the image enlarge.

![1](https://user-images.githubusercontent.com/43103079/71713298-80b59f00-2e4c-11ea-8737-bb120cce2e6f.png)

Below is a table how to control the smile face object.

![2](https://user-images.githubusercontent.com/43103079/71713299-80b59f00-2e4c-11ea-8843-4158f98667bf.png)

In CV_A2/A2_homography.py, the orb matcher is implemented. 

![3](https://user-images.githubusercontent.com/43103079/71713301-80b59f00-2e4c-11ea-9efa-0d95e7d28858.png)

Then, it performs image matching with two method, normalization and RANSAC.


Finally, it is used to make a panorama image by two images. This program find similar features in image and connect it automatically.


Moreover, it interpolate the two image.

