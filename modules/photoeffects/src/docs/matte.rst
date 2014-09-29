=======================================
Matte
=======================================
Increases the brightness of peripheral pixels.

.. cpp:function:: int matte(cv::InputArray src, cv::OutputArray dst, cv::Point firstPoint, cv::Point secondPoint, float sigma)

   :param src: RGB image.
   :param dst: Destination image of the same size and the same type as **src**.
   :param firstPoint: The first point for creating ellipse.
   :param secondPoint: The second point for creatin ellipse.
   :param sigma: The deviation in the Gaussian blur effect.
   :return: Error code.

The algorithm.

#. Create new image with white background for mask.
#. Draw black ellipse inscribed in a rectangle that is defined by two opposite corner points (**firstPoint** and **secondPoint**) on the mask image. It's a meaning part.
#. Apply gaussian blur to the meaning part to make fade effect.
#. Convolve mask with the image.
#. Convert resulting image to the same color format as **src**.

Example.

    sigma = 25.

|srcImage| |dstImage|

.. |srcImage| image:: matte_before.jpg
   :width: 40%

.. |dstImage| image:: matte_after.jpg
   :width: 40%
