=======================================
Fade Color
=======================================
Applies color fade effect to image.

.. cpp:function:: void fadeColor(InputArray src, OutputArray dst, Point startPoint, Point endPoint)

   :param src: Grayscale or RGB image.
   :param dst: Destination image of the same size and the same type as **src**.
   :param startPoint: Initial point of direction vector for color fading.
   :param endPoint: Terminal point of direction vector for color fading.

The algorithm.

1. Determine the coordinates of the vector by two points **(startPoint, endPoint)** .
2. Determine the line which is perpendicular to vector and is passing through **startPoint**.
3. Find the most distant point from the line.
4. For each pixel located at one side from the line defined by the direction of the vector, change the value of each channel by the following formula:

        **newValue = (1-a) * oldValue + a * 255**, a = distance / maxDistance.

5. Save this matrix as image in same format.


Example.

|srcImage| |dstImage|

.. |srcImage| image:: pics/fade_color_before.jpg
   :width: 40%

.. |dstImage| image:: pics/fade_color_after.jpg
   :width: 40%
