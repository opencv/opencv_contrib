=========================================
Warmify
=========================================

Increases saturation of red and yellow tones, making photographs more warm, sunset view.

.. cpp:function:: int warmify(cv::InputArray src, cv::OutputArray dst, uchar delta = 30)

   :param src: Source 8-bit three-channel image.
   :param dst: Destination image of the same size and the same type as **src**.
   :param delta: Value by which saturation of warm colors is increased.
   :return: Error code.

The algorithm.

#. Create 3-channel image, which is interpreted as BGR image.

    #. 1st channel is the matrix, each element equals **blue** = blue_src.
    #. 2nd channel is the matrix, each element equals **green** = green_src + delta.
    #. 3rd channel is the matrix, each element equals **red** = red_src + delta.

#. Save this matrix as BGR image.

Example.

|srcImage| |dstImage|

.. |srcImage| image:: warmify_before.jpg
   :width: 40%

.. |dstImage| image:: warmify_after.jpg
   :width: 40%
