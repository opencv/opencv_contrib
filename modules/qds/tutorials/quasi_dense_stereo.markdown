Quasi dense Stereo {#quasi_dense_stereo}
==================

Goal
----

In this tutorial you will learn how to

-   Configure a QuasiDenseStero object
-   Compute dense Stereo correspondences.

-----------
[Source Code](../samples/dense_disparity.cpp)

## Explanation:

The program loads a stereo image pair.


After importing the images.
```
Mat rightImg, leftImg;
leftImg = imread("./imgLeft.png", IMREAD_COLOR);
rightImg = imread("./imgRight.png", IMREAD_COLOR);
```
We need to know the frame size of a single image, in order to create an instance of a `QuasiDesnseStereo` object.
```
cv::Size frameSize = leftImg.size();
qds::QuasiDenseStereo stereo(frameSize);
```
Because we didn't specify the second argument in the constructor, the `QuasiDesnseStereo` object will
load default parameters from this [headerfile](../include/opencv2/qds/defaults.hpp).

We can then pass the imported stereo images in the process method like this
```
stereo.process(leftChannel, rightChannel);
```
The process method contains most of the functionality of the class and does two main things.
-   Computes a sparse stereo based in "Good Features to Track" and "pyramidal Lucas-Kanade" flow algorithm
-   Based on those sparse stereo points, densifies the stereo correspondences using Quasi Dense Stereo method.

After the execution of `process()` we can display the disparity Image of the stereo.
```
int displvl = 80;
Mat disp;
disp = stereo.getDisparity(displvl);
cv::namedWindow("disparity map");
cv::imshow("disparity map", disp);
```

At this point we can also extract all the corresponding points using `getDenseMatches()` method.
```
vector<qds::Match> matches;
stereo.getDenseMatches(matches);
```
