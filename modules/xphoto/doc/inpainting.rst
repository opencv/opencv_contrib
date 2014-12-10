Single image inpainting
***********************

.. highlight:: cpp

Inpainting
----------
.. ocv:function:: void inpaint(const Mat &src, const Mat &mask, Mat &dst, const int algorithmType)

	The function implements different single-image inpainting algorithms.

    :param src: source image, it could be of any type and any number of channels from 1 to 4. In case of 3- and 4-channels images the function expect them in CIELab colorspace or similar one, where first color component shows intensity, while second and third shows colors. Nonetheless you can try any colorspaces.
    :param mask: mask (CV_8UC1), where non-zero pixels indicate valid image area, while zero pixels indicate area to be inpainted
    :param dst: destination image
    :param algorithmType: expected noise standard deviation
        * INPAINT_SHIFTMAP: This algorithm searches for dominant correspondences (transformations) of image patches and tries to seamlessly fill-in the area to be inpainted using this transformations. Look in the original paper [He2012]_ for details.

    .. [He2012] K. He, J. Sun., "Statistics of Patch Offsets for Image Completion",
                IEEE European Conference on Computer Vision (ICCV), 2012,
                pp. 16-29. `DOI <http://dx.doi.org/10.1007/978-3-642-33709-3_2>`_