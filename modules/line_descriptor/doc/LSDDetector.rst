.. _LSDDetector:

Line Segments Detector
======================


Lines extraction methodology
----------------------------

The lines extraction methodology described in the following is mainly based on [EDLN]_.
The extraction starts with a Gaussian pyramid generated from an original image, downsampled N-1 times, blurred N times, to obtain N layers (one for each octave), with layer 0 corresponding to input image. Then, from each layer (octave) in the pyramid, lines are extracted using LSD algorithm. 

Differently from EDLine lines extractor used in original article, LSD furnishes information only about lines extremes; thus, additional information regarding slope and equation of line are computed via analytic methods. The number of pixels is obtained using *LineIterator*. Extracted lines are returned in the form of KeyLine objects, but since extraction is based on a method different from the one used in *BinaryDescriptor* class, data associated to a line's extremes in original image and in octave it was extracted from, coincide. KeyLine's field *class_id* is used as an index to indicate the order of extraction of a line inside a single octave.


LSDDetector::createLSDDetector
------------------------------

Creates ad LSDDetector object, using smart pointers.

.. ocv:function:: Ptr<LSDDetector> LSDDetector::createLSDDetector()


LSDDetector::detect
-------------------

Detect lines inside an image.

.. ocv:function:: void LSDDetector::detect( const Mat& image, std::vector<KeyLine>& keylines, int scale, int numOctaves, const Mat& mask=Mat()) 

.. ocv:function:: void LSDDetector::detect( const std::vector<Mat>& images, std::vector<std::vector<KeyLine> >& keylines, int scale, int numOctaves, const std::vector<Mat>& masks=std::vector<Mat>() ) const

	:param image: input image

	:param images: input images

	:param keylines: vector or set of vectors that will store extracted lines for one or more images

	:param mask: mask matrix to detect only KeyLines of interest

	:param masks: vector of mask matrices to detect only KeyLines of interest from each input image

	:param scale: scale factor used in pyramids generation

	:param numOctaves: number of octaves inside pyramid


References
----------

.. [EDLN] Von Gioi, R. Grompone, et al. *LSD: A fast line segment detector with a false detection control*, IEEE Transactions on Pattern Analysis and Machine Intelligence 32.4 (2010): 722-732.

