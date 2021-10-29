Structured forests for fast edge detection {#tutorial_ximgproc_prediction}
==========================================

Introduction
------------

In this tutorial you will learn how to use structured forests for the purpose of edge detection in
an image.

Examples
--------

![image](images/01.jpg)

![image](images/02.jpg)

![image](images/03.jpg)

![image](images/04.jpg)

![image](images/05.jpg)

![image](images/06.jpg)

![image](images/07.jpg)

![image](images/08.jpg)

![image](images/09.jpg)

![image](images/10.jpg)

![image](images/11.jpg)

![image](images/12.jpg)

@note binarization techniques like Canny edge detector are applicable to edges produced by both
algorithms (Sobel and StructuredEdgeDetection::detectEdges).

Source Code
-----------

@includelineno ximgproc/samples/structured_edge_detection.cpp

Explanation
-----------

-#  **Load source color image**

    @snippet ximgproc/samples/structured_edge_detection.cpp imread

-#  **Convert source image to float [0;1] range**

    @snippet ximgproc/samples/structured_edge_detection.cpp convert

-#  **Run main algorithm**

    @snippet ximgproc/samples/structured_edge_detection.cpp create
    @snippet ximgproc/samples/structured_edge_detection.cpp detect
    @snippet ximgproc/samples/structured_edge_detection.cpp nms

-#  **Show results**

    @snippet ximgproc/samples/structured_edge_detection.cpp imshow

Literature
----------

For more information, refer to the following papers : @cite Dollar2013 @cite Lim2013
