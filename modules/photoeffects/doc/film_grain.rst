=======================================
Film Grain
=======================================
Applies film grain effect to the initial image.

.. cpp:function:: void filmGrain(InputArray src, OutputArray dst, int grainValue, int seed)

   :param src: Grayscale or RGB image.
   :param dst: Destination image of the same size and the same type as **src**.
   :param grainValue: Degree of graininess. 8 is default value.
   :param seed: Number used to initialize a random number generator. 0 is default value.

The algorithm.

1. Create matrix with noise.
2. Add noise to image.

Example.

|srcImage| |dstImage|

.. |srcImage| image:: pics/film_grain_before.png
   :width: 40%

.. |dstImage| image:: pics/film_grain_after.png
   :width: 40%
