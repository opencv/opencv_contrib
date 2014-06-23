.. ximgproc:

Structured forests for fast edge detection
******************************************

Introduction
------------
Today most digital images and imaging devices use 8 bits per channel thus limiting the dynamic range of the device to two orders of magnitude (actually 256 levels), while human eye can adapt to lighting conditions varying by ten orders of magnitude. When we take photographs of a real world scene bright regions may be overexposed, while the dark ones may be underexposed, so we can’t capture all details using a single exposure. HDR imaging works with images that use more that 8 bits per channel (usually 32-bit float values), allowing much wider dynamic range.

There are different ways to obtain HDR images, but the most common one is to use photographs of the scene taken with different exposure values. To combine this exposures it is useful to know your camera’s response function and there are algorithms to estimate it. After the HDR image has been blended it has to be converted back to 8-bit to view it on usual displays. This process is called tonemapping. Additional complexities arise when objects of the scene or camera move between shots, since images with different exposures should be registered and aligned.

In this tutorial we show how to generate and display HDR image from an exposure sequence. In our case images are already aligned and there are no moving objects. We also demonstrate an alternative approach called exposure fusion that produces low dynamic range image. Each step of HDR pipeline can be implemented using different algorithms so take a look at the reference manual to see them all.

Examples
--------

.. image:: images/01.jpg
  :height: 238pt
  :width:  750pt
  :alt: First example
  :align: center

.. image:: images/02.jpg
  :height: 238pt
  :width:  750pt
  :alt: First example
  :align: center

.. image:: images/03.jpg
  :height: 238pt
  :width:  750pt
  :alt: First example
  :align: center

.. image:: images/04.jpg
  :height: 238pt
  :width:  750pt
  :alt: First example
  :align: center

.. image:: images/05.jpg
  :height: 238pt
  :width:  750pt
  :alt: First example
  :align: center

.. image:: images/06.jpg
  :height: 238pt
  :width:  750pt
  :alt: First example
  :align: center

.. image:: images/07.jpg
  :height: 238pt
  :width:  750pt
  :alt: First example
  :align: center

.. image:: images/08.jpg
  :height: 238pt
  :width:  750pt
  :alt: First example
  :align: center

.. image:: images/09.jpg
  :height: 238pt
  :width:  750pt
  :alt: First example
  :align: center

.. image:: images/10.jpg
  :height: 238pt
  :width:  750pt
  :alt: First example
  :align: center

.. image:: images/11.jpg
  :height: 238pt
  :width:  750pt
  :alt: First example
  :align: center

.. image:: images/12.jpg
  :height: 238pt
  :width:  750pt
  :alt: First example
  :align: center

**Note :** binarization techniques like Canny edge detector are applicable
           to edges produced by both algorithms (``Sobel`` and ``StructuredEdgeDetection::detectEdges``).

Source Code
-----------

.. literalinclude:: ../../../../modules/ximpgroc/samples/cpp/structured_edge_detection.cpp
   :language: cpp
   :linenos:
   :tab-width: 4

Explanation
-----------

1. **Load source color image**

  .. code-block:: cpp

    cv::Mat image = cv::imread(inFilename, 1);
    if ( image.empty() )
    {
        printf("Cannot read image file: %s\n", inFilename.c_str());
        return -1;
    }

2. **Convert source image to [0;1] range and RGB colospace**

  .. code-block:: cpp

    cv::cvtColor(image, image, CV_BGR2RGB);
    image.convertTo(image, cv::DataType<float>::type, 1/255.0);

3. **Run main algorithm**

  .. code-block:: cpp

    cv::Mat edges(image.size(), image.type());

    cv::Ptr<StructuredEdgeDetection> pDollar =
        cv::createStructuredEdgeDetection(modelFilename);
    pDollar->detectEdges(image, edges);

4. **Show results**

  .. code-block:: cpp

    if ( outFilename == "" )
    {
        cv::namedWindow("edges", 1);
        cv::imshow("edges", edges);

        cv::waitKey(0);
    }
    else
        cv::imwrite(outFilename, 255*edges);

Literature
----------
For more information, refer to the following papers :

.. [Dollar2013] Dollar P., Zitnick C. L., "Structured forests for fast edge detection",
                IEEE International Conference on Computer Vision (ICCV), 2013,
                pp. 1841-1848. `DOI <http://dx.doi.org/10.1109/ICCV.2013.231>`_

.. [Lim2013] Lim J. J., Zitnick C. L., Dollar P., "Sketch Tokens: A Learned
             Mid-level Representation for Contour and Object Detection",
             Comoputer Vision and Pattern Recognition (CVPR), 2013,
             pp. 3158-3165. `DOI <http://dx.doi.org/10.1109/CVPR.2013.406>`_
