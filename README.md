PolarCalibration
================

Polar Calibration method based on the paper "M. Pollefeys, R. Koch and L. Van Gool, A simple and efficient rectification method for general motion, Proc. International Conference on Computer Vision, pp.496-501, Corfu (Greece), 1999.", which is available at http://www.inf.ethz.ch/personal/pomarc/pubs/PollefeysICCV99.pdf

The idea here is not just to align left-right images, but being able to align images between frames. So the goal here is being able to align pairs of images between frames, even if they are taken by a moving camera. Having this alignment, it is possible to, for example, detect changes between frames.

Usage
-----
You just need to create a new object of the class PolarCalibration and call the method compute. You can do it in three different ways:

- [compute(const cv::Mat& img1distorted, const cv::Mat& img2distorted, 
                               const cv::Mat & cameraMatrix1, const cv::Mat & distCoeffs1, 
                               const cv::Mat & cameraMatrix2, const cv::Mat & distCoeffs2)](https://github.com/nestormh/PolarCalibration/blob/master/src/polarcalibration.cpp#L39):

In this, you pass the distorted images, and the rectification parameters. Then, the code will undistort the image, compute the correspondences and the Fundamental matrix, and after that you will be able to retrieve the rectified images.

- [compute(const cv::Mat& img1, const cv::Mat& img2, cv::Mat F, 
                               vector< cv::Point2f > points1, vector< cv::Point2f > points2, const uint32_t method)](https://github.com/nestormh/PolarCalibration/blob/master/src/polarcalibration.cpp#L54):

In this, you just pass the undistorted images.
Optionally, you can pass also the Fundamental Matrix and a set of computed correspondences. If you just pass to the function the Undistorted images, it will automatically compute the correspondences.

- An example of how you can use this code is shown [here](https://github.com/nestormh/PolarCalibration/blob/master/src/main.cpp#L56)

- You might also want to modify the code in ([here](https://github.com/nestormh/PolarCalibration/blob/master/src/polarcalibration.cpp#L39), or [here](https://github.com/nestormh/PolarCalibration/blob/master/src/polarcalibration.cpp#L54) ) in order to get a comfortable way to get the results you want.

- Using method [getRectifiedImages](https://github.com/nestormh/PolarCalibration/blob/master/src/polarcalibration.cpp#L1059), you can retrieve the aligned images once it the computation process is finished.

Issues
------

If rectification fails, I suggest you to check the correspondences between images, as well as the Fundamental Matrix being passed.

Please notice
-------------
- This code is far from being properly tested, it was created just for research purposes and I haven't used it for a while.
- This code is also far from being optimized.

However, if you just want to get some results or you are not interested in speed, it can be helpful, or a good starting point.
