=======================================
Edge Blur
=======================================
Blurs the edges of an image, keeping center in focus.

.. cpp:function:: int edgeBlur(InputArray src, OutputArray dst, int indentTop, int indentLeft)

   :param src: RGB image.
   :param dst: Destination image of the same size and the same type as **src**.
   :param indentTop: The indent from the top and the bottom of the image. It will be blured.
   :param indentLeft: The indent from the left and right side of the image. It will be blured.
   :return: Error code.

The algorithm.

    The amount of blurring is defined by the gaussian filter's kernel size and standard deviation and depends on the weighted distance from the center of the image:

    .. math::
       d(x, y) = \frac{(x - a)^2}{(a - indentLeft)^2} + \frac{(y - b)^2}{(b - indentTop)^2},

    where :math:`a = \frac{src_{width}}{2}, b = \frac{src_{height}}{2}`. For each pixel :math:`(x, y)` of the image, if the distance :math:`d(x, y)` is greater than 1, gaussian filter with center at :math:`(x,y)`, kernel size :math:`d(x, y)` and standard deviation :math:`(radius - 0.5)` is applied, otherwise **src** image pixel is left unchanged. Border pixels are replicated to fit the kernel size.


Example.

    **indentTop** = 90, **indentLeft** = 90.

|src| |dst|

.. |src| image:: edge_blur_before.png
   :width: 40%

.. |dst| image:: edge_blur_after.png
   :width: 40%