==========================
Antique
==========================
Applies antique effect to image.

.. cpp:function:: int antique(cv::InputArray src, cv::InputArray texture, cv::OutputArray dst)

   :param src: RGB image.
   :param texture: RGB image that is overlaid with the **src** image as a texture of an old photo.
   :param dst: Destination image of the same size and the same type as **src**.
   :return: Error code.

The algorithm.

#. Resize texture to match the source image size.
#. Apply sepia matrix transform to the **src**.
#. Calculate the weighted sum of the **texture** and **src**.
#. Convert resulting image to the same color format as **src**.

Example.

|srcImage| |dstImage|

.. |srcImage| image:: antique_before.jpg
   :width: 40%

.. |dstImage| image:: antique_after.jpg
   :width: 40%
