Oil painting effect {#tutorial_xphoto_oil_painting_effect}
===================================================

Introduction
------------
Image is converted in a color space default color space COLOR_BGR2GRAY.
For every pixel in the image a program calculated a histogram (first plane of color space) of the neighbouring of size 2*size+1.
and assigned the value of the most frequently occurring value. The result looks almost like an oil painting. Parameter 4 of oilPainting is used to decrease image dynamic and hence increase oil painting effect.

Example
--------------------


    @code{.cpp}
    Mat img;
    Mat dst;
    img = imread("opencv/samples/data/baboon.jpg");
    xphoto::oilPainting(img, dst, 10, 1, COLOR_BGR2Lab);
    imshow("oil painting effect", dst);
    @endcode

    Original ![](images/baboon.jpg)
    Oil painting effect ![](images/baboon_oil_painting_effect.jpg)
