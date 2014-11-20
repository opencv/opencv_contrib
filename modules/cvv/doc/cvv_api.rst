CVV : the API
*************

.. highlight:: cpp


Introduction
++++++++++++

Namespace for all functions is **cvv**, i.e. *cvv::showImage()*.

Compilation:

* For development, i.e. for cvv GUI to show up, compile your code using cvv with *g++ -DCVVISUAL_DEBUGMODE*.
* For release, i.e. cvv calls doing nothing, compile your code without above flag. 

See cvv tutorial for a commented example application using cvv.




API Functions
+++++++++++++


showImage
---------
Add a single image to debug GUI (similar to :imshow:`imshow <>`).

.. ocv:function:: void showImage(InputArray img, const CallMetaData& metaData, const string& description, const string& view)

    :param img: Image to show in debug GUI.
    :param metaData: Properly initialized CallMetaData struct, i.e. information about file, line and function name for GUI. Use CVVISUAL_LOCATION macro.
    :param description: Human readable description to provide context to image.
    :param view: Preselect view that will be used to visualize this image in GUI. Other views can still be selected in GUI later on.



debugFilter
-----------
Add two images to debug GUI for comparison. Usually the input and output of some filter operation, whose result should be inspected.

.. ocv:function:: void debugFilter(InputArray original, InputArray result, const CallMetaData& metaData, const string& description, const string& view)

    :param original: First image for comparison, e.g. filter input.
    :param result: Second image for comparison, e.g. filter output.
    :param metaData: See :ocv:func:`showImage`
    :param description: See :ocv:func:`showImage`
    :param view: See :ocv:func:`showImage`



debugDMatch
-----------
Add a filled in :basicstructures:`DMatch <dmatch>` to debug GUI. The matches can are visualized for interactive inspection in different GUI views (one similar to an interactive :draw_matches:`drawMatches<>`).

.. ocv:function:: void debugDMatch(InputArray img1, std::vector<cv::KeyPoint> keypoints1, InputArray img2, std::vector<cv::KeyPoint> keypoints2, std::vector<cv::DMatch> matches, const CallMetaData& metaData, const string& description, const string& view, bool useTrainDescriptor)

    :param img1: First image used in :basicstructures:`DMatch <dmatch>`.
    :param keypoints1: Keypoints of first image.
    :param img2:  Second image used in DMatch.
    :param keypoints2:  Keypoints of second image.
    :param metaData: See :ocv:func:`showImage`
    :param description: See :ocv:func:`showImage`
    :param view: See :ocv:func:`showImage`
    :param useTrainDescriptor: Use :basicstructures:`DMatch <dmatch>`'s train descriptor index instead of query descriptor index.



finalShow
---------
This function **must** be called *once* *after* all cvv calls if any.
As an alternative create an instance of FinalShowCaller, which calls finalShow() in its destructor (RAII-style).

.. ocv:function:: void finalShow()



setDebugFlag
------------
Enable or disable cvv for current translation unit and thread (disabled this way has higher - but still low - overhead compared to using the compile flags).

.. ocv:function:: void setDebugFlag(bool active)

    :param active: See above
