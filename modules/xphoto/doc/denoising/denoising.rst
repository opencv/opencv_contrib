Image denoising techniques
**************************

.. highlight:: cpp

dctDenoising
------------
.. ocv:function:: void dctDenoising(const Mat &src, Mat &dst, const float sigma)

	The function implements simple dct-based denoising,
	link: http://www.ipol.im/pub/art/2011/ys-dct/.

    :param src: source image
    :param dst: destination image
    :param sigma: expected noise standard deviation
    :param psize: size of block side where dct is computed

.. seealso::

    :ocv:func:`fastNlMeansDenoising`
