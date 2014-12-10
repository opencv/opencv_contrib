Automatic white balance correction
**********************************

.. highlight:: cpp

balanceWhite
------------

.. ocv:function:: void balanceWhite(const Mat &src, Mat &dst, const int algorithmType, const float inputMin  = 0.0f, const float inputMax  = 255.0f, const float outputMin = 0.0f, const float outputMax = 255.0f)

	The function implements different algorithm of automatic white balance, i.e. it tries to map image's white color to perceptual white (this can be violated due to specific illumination or camera settings).

    :param algorithmType: type of the algorithm to use. Use WHITE_BALANCE_SIMPLE to perform smart histogram adjustments (ignoring 4% pixels with minimal and maximal values) for each channel.
    :param inputMin: minimum value in the input image
    :param inputMax: maximum value in the input image
    :param outputMin: minimum value in the output image
    :param outputMax: maximum value in the output image

.. seealso::

    :ocv:func:`cvtColor`,
    :ocv:func:`equalizeHist`
