=========================================
Warmify
=========================================

Increases saturation of red and yellow tones, making photographs more warm, sunset view.

.. ocv:function:: void warmify(cv::InputArray src, cv::OutputArray dst, uchar delta = 30)

   :param src: Source 3-channel image.
   :param dst: Destination image of the same size and the same type as **src**.
   :param delta: Value by which saturation of warm colors is increased.

The algorithm.

 For every pixel of the **dst** image:

  #. 1st channel is *blue = blue_src*.
  #. 2nd channel is *green = green_src + delta*.
  #. 3rd channel is *red = red_src + delta*.

Example.

|srcImage| |dstImage|

.. |srcImage| image:: pics/warmify_before.jpg
   :width: 40%

.. |dstImage| image:: pics/warmify_after.jpg
   :width: 40%
