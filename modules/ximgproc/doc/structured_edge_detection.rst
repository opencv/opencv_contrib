Structured forests for fast edge detection
******************************************

.. highlight:: cpp

This module contains implementations of modern structured edge detection algorithms,
i.e. algorithms which somehow takes into account pixel affinities in natural images.

StructuredEdgeDetection
-----------------------

.. ocv:class:: StructuredEdgeDetection : public Algorithm

Class implementing edge detection algorithm from [Dollar2013]_ ::

    /*! \class StructuredEdgeDetection
    Prediction part of [P. Dollar and C. L. Zitnick. Structured Forests for Fast Edge Detection, 2013].
    */
    class CV_EXPORTS_W StructuredEdgeDetection : public Algorithm
    {
    public:

        /*!
        * The function detects edges in src and draw them to dst
        *
        * \param src : source image (RGB, float, in [0;1]) to detect edges
        * \param dst : destination image (grayscale, float, in [0;1])
        *              where edges are drawn
        */
        CV_WRAP virtual void detectEdges(const Mat src, Mat dst) = 0;
    };

    /*!
    * The only available constructor loading data from model file
    *
    * \param model : name of the file where the model is stored
    */
    CV_EXPORTS_W Ptr<StructuredEdgeDetection> createStructuredEdgeDetection(const String &model);

StructuredEdgeDetection::detectEdges
++++++++++++++++++++++++++++++++++++
.. ocv:function:: void detectEdges(const Mat src, Mat dst)

    The function detects edges in src and draw them to dst. The algorithm underlies this function
    is much more robust to texture presence, than common approaches, e.g. Sobel

    :param src: source image (RGB, float, in [0;1]) to detect edges
    :param dst: destination image (grayscale, float, in [0;1])
                where edges are drawn

.. seealso::

    :ocv:func:`Sobel`,
    :ocv:func:`Canny`

createStructuredEdgeDetection
+++++++++++++++++++++++++++++
.. ocv:function:: Ptr<cv::StructuredEdgeDetection> createStructuredEdgeDetection(String model)

    The only available constructor

    :param model: model file name


.. [Dollar2013] P. Dollár, C. L. Zitnick, "Structured forests for fast edge detection",
                IEEE International Conference on Computer Vision (ICCV), 2013,
                pp. 1841-1848. `DOI <http://dx.doi.org/10.1109/ICCV.2013.231>`_
