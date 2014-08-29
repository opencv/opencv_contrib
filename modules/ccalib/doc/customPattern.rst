Custom Calibration Pattern
==========================

.. highlight:: cpp

CustomPattern
-------------
A custom pattern class that can be used to calibrate a camera and to further track the translation and rotation of the pattern. Defaultly it uses an ``ORB`` feature detector and a ``BruteForce-Hamming(2)`` descriptor matcher to find the location of the pattern feature points that will subsequently be used for calibration.

.. ocv:class:: CustomPattern : public Algorithm


CustomPattern::CustomPattern
----------------------------
CustomPattern constructor.

.. ocv:function:: CustomPattern()


CustomPattern::create
---------------------
A method that initializes the class and generates the necessary detectors, extractors and matchers.

.. ocv:function:: bool create(InputArray pattern, const Size2f boardSize, OutputArray output = noArray())

    :param pattern: The image, which will be used as a pattern. If the desired pattern is part of a bigger image, you can crop it out using image(roi).

    :param boardSize: The size of the pattern in physical dimensions. These will be used to scale the points when the calibration occurs.

    :param output: A matrix that is the same as the input pattern image, but has all the feature points drawn on it.

    :return Returns whether the initialization was successful or not. Possible reason for failure may be that no feature points were detected.

.. seealso::

    :ocv:func:`getFeatureDetector`,
    :ocv:func:`getDescriptorExtractor`,
    :ocv:func:`getDescriptorMatcher`

.. note::

   * Determine the number of detected feature points can be done through :ocv:func:`getPatternPoints` method.

   * The feature detector, extractor and matcher cannot be changed after initialization.



CustomPattern::findPattern
--------------------------
Finds the pattern in the input image

.. ocv:function:: bool findPattern(InputArray image, OutputArray matched_features, OutputArray pattern_points, const double ratio = 0.7, const double proj_error = 8.0, const bool refine_position = false, OutputArray out = noArray(), OutputArray H = noArray(), OutputArray pattern_corners = noArray());

    :param image: The input image where the pattern is searched for.

    :param matched_features: A ``vector<Point2f>`` of the projections of calibration pattern points, matched in the image. The points correspond to the ``pattern_points``.``matched_features`` and ``pattern_points`` have the same size.

    :param pattern_points: A ``vector<Point3f>`` of calibration pattern points in the calibration pattern coordinate space.

    :param ratio: A ratio used to threshold matches based on D. Lowe's point ratio test.

    :param proj_error: The maximum projection error that is allowed when the found points are back projected. A lower projection error will be beneficial for eliminating mismatches. Higher values are recommended when the camera lens has greater distortions.

    :param refine_position: Whether to refine the position of the feature points with :ocv:func:`cornerSubPix`.

    :param out: An image showing the matched feature points and a contour around the estimated pattern.

    :param H: The homography transformation matrix between the pattern and the current image.

    :param pattern_corners: A ``vector<Point2f>`` containing the 4 corners of the found pattern.

    :return The method return whether the pattern was found or not.


CustomPattern::isInitialized
----------------------------

.. ocv:function:: bool isInitialized()

    :return If the class is initialized or not.


CustomPattern::getPatternPoints
-------------------------------

.. ocv:function:: void getPatternPoints(OutputArray original_points)

    :param original_points: Fills the vector with the points found in the pattern.


CustomPattern::getPixelSize
---------------------------
.. ocv:function:: double getPixelSize()

    :return Get the physical pixel size as initialized by the pattern.


CustomPattern::setFeatureDetector
---------------------------------
 .. ocv:function:: bool setFeatureDetector(Ptr<FeatureDetector> featureDetector)

    :param featureDetector: Set a new FeatureDetector.

    :return Is it successfully set? Will fail if the object is already initialized by :ocv:func:`create`.

.. note::

    * It is left to user discretion to select matching feature detector, extractor and matchers. Please consult the documentation for each to confirm coherence.


CustomPattern::setDescriptorExtractor
-------------------------------------
.. ocv:function:: bool setDescriptorExtractor(Ptr<DescriptorExtractor> extractor)

    :param extractor: Set a new DescriptorExtractor.

    :return Is it successfully set? Will fail if the object is already initialized by :ocv:func:`create`.


CustomPattern::setDescriptorMatcher
-----------------------------------
.. ocv:function:: bool setDescriptorMatcher(Ptr<DescriptorMatcher> matcher)

    :param matcher: Set a new DescriptorMatcher.

    :return Is it successfully set? Will fail if the object is already initialized by :ocv:func:`create`.


CustomPattern::getFeatureDetector
---------------------------------
.. ocv:function:: Ptr<FeatureDetector> getFeatureDetector()

    :return The used FeatureDetector.


CustomPattern::getDescriptorExtractor
-------------------------------------
.. ocv:function:: Ptr<DescriptorExtractor> getDescriptorExtractor()

    :return The used DescriptorExtractor.


CustomPattern::getDescriptorMatcher
-----------------------------------
.. ocv:function:: Ptr<DescriptorMatcher> getDescriptorMatcher()

    :return The used DescriptorMatcher.


CustomPattern::calibrate
------------------------
Calibrates the camera.

.. ocv:function:: double calibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, Size imageSize, InputOutputArray cameraMatrix, InputOutputArray distCoeffs, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, int flags = 0, TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON))

    See :ocv:func:`calibrateCamera` for parameter information.


CustomPattern::findRt
---------------------
Finds the rotation and translation vectors of the pattern.

.. ocv:function:: bool findRt(InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess = false, int flags = ITERATIVE)
.. ocv:function:: bool findRt(InputArray image, InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess = false, int flags = ITERATIVE)

    :param image: The image, in which the rotation and translation of the pattern will be found.

    See :ocv:func:`solvePnP` for parameter information.


CustomPattern::findRtRANSAC
---------------------------
Finds the rotation and translation vectors of the pattern using RANSAC.

.. ocv:function:: bool findRtRANSAC(InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess = false, int iterationsCount = 100, float reprojectionError = 8.0, int minInliersCount = 100, OutputArray inliers = noArray(), int flags = ITERATIVE)
.. ocv:function:: bool findRtRANSAC(InputArray image, InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess = false, int iterationsCount = 100, float reprojectionError = 8.0, int minInliersCount = 100, OutputArray inliers = noArray(), int flags = ITERATIVE)

    :param image: The image, in which the rotation and translation of the pattern will be found.

    See :ocv:func:`solvePnPRANSAC` for parameter information.


CustomPattern::drawOrientation
------------------------------
Draws the ``(x,y,z)`` axis on the image, in the center of the pattern, showing the orientation of the pattern.

.. ocv:function:: void drawOrientation(InputOutputArray image, InputArray tvec, InputArray rvec, InputArray cameraMatrix, InputArray distCoeffs, double axis_length = 3, int axis_width = 2)

    :param image: The image, based on which the rotation and translation was calculated. The axis will be drawn in color - ``x`` - in red, ``y`` - in green, ``z`` - in blue.

    :param tvec: Translation vector.

    :param rvec: Rotation vector.

    :param cameraMatrix: The camera matrix.

    :param distCoeffs: The distortion coefficients.

    :param axis_length: The length of the axis symbol.

    :param axis_width: The width of the axis symbol.

