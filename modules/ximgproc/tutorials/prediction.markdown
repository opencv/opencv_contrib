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
    @code{.cpp}
    cv::Mat image = cv::imread(inFilename, 1);
    if ( image.empty() )
    {
        printf("Cannot read image file: %s\n", inFilename.c_str());
        return -1;
    }
    @endcode

-#  **Convert source image to [0;1] range**
    @code{.cpp}
    image.convertTo(image, cv::DataType<float>::type, 1/255.0);
    @endcode

-#  **Run main algorithm**
    @code{.cpp}
    cv::Mat edges(image.size(), image.type());

    cv::Ptr<StructuredEdgeDetection> pDollar =
        cv::createStructuredEdgeDetection(modelFilename);
    pDollar->detectEdges(image, edges);
    @endcode

-#  **Show results**
    @code{.cpp}
    if ( outFilename == "" )
    {
        cv::namedWindow("edges", 1);
        cv::imshow("edges", edges);

        cv::waitKey(0);
    }
    else
        cv::imwrite(outFilename, 255*edges);
    @endcode

Literature
----------

For more information, refer to the following papers : @cite Dollar2013 @cite Lim2013
