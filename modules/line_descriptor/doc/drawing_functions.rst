.. _drawing:

Drawing Functions for Keylines and Matches
==========================================

.. highlight:: cpp

drawLineMatches
---------------

Draws the found matches of keylines from two images.

.. ocv:function:: void drawLineMatches( const Mat& img1, const std::vector<KeyLine>& keylines1, const Mat& img2, const std::vector<KeyLine>& keylines2, const std::vector<DMatch>& matches1to2, Mat& outImg, const Scalar& matchColor=Scalar::all(-1), const Scalar& singleLineColor=Scalar::all(-1), const std::vector<char>& matchesMask=std::vector<char>(), int flags=DrawLinesMatchesFlags::DEFAULT )

	:param img1: first image
	:param keylines1: keylines extracted from first image
	:param img2: second image
	:param keylines2: keylines extracted from second image
	:param matches1to2: vector of matches
	:param outImg: output matrix to draw on
	:param matchColor: drawing color for matches (chosen randomly in case of default value)
	:param singleLineColor: drawing color for keylines (chosen randomly in case of default value)
	:param matchesMask: mask to indicate which matches must be drawn
	:param flags: drawing flags

.. note:: If both *matchColor* and *singleLineColor* are set to their default values, function draws matched lines and line connecting them with same color

The structure of drawing flags is shown in the following:

.. code-block:: cpp

	/* struct for drawing options */
	struct CV_EXPORTS DrawLinesMatchesFlags
	{
		enum
		{
		    DEFAULT = 0, // Output image matrix will be created (Mat::create),
		                 // i.e. existing memory of output image may be reused.
		                 // Two source images, matches, and single keylines
		                 // will be drawn.
		    DRAW_OVER_OUTIMG = 1, // Output image matrix will not be
		                   // created (using Mat::create). Matches will be drawn
		                   // on existing content of output image.
		    NOT_DRAW_SINGLE_LINES = 2 // Single keylines will not be drawn.
		};
	};

..


drawKeylines
------------

Draws keylines. 

.. ocv:function:: void drawKeylines( const Mat& image, const std::vector<KeyLine>& keylines, Mat& outImage, const Scalar& color=Scalar::all(-1), int flags=DrawLinesMatchesFlags::DEFAULT )

	:param image: input image
	:param keylines: keylines to be drawn
	:param outImage: output image to draw on
	:param color: color of lines to be drawn (if set to defaul value, color is chosen randomly)
	:param flags: drawing flags


Related pages
-------------

* :ref:`line_descriptor`
* :ref:`binary_descriptor`
* :ref:`matching`
