.. highlight:: cpp

Domain Transform filter
====================================

This section describes interface for Domain Transform filter.
For more details about this filter see [Gastal11]_ and References_.

DTFilter
------------------------------------
.. ocv:class:: DTFilter : public Algorithm

Interface for realizations of Domain Transform filter.

createDTFilter
------------------------------------
Factory method, create instance of :ocv:class:`DTFilter` and produce initialization routines.

.. ocv:function:: Ptr<DTFilter> createDTFilter(InputArray guide, double sigmaSpatial, double sigmaColor, int mode = DTF_NC, int numIters = 3)

.. ocv:pyfunction:: cv2.createDTFilter(guide, sigmaSpatial, sigmaColor[, mode[, numIters]]) -> instance

    :param guide: guided image (used to build transformed distance, which describes edge structure of guided image).
    :param sigmaSpatial: :math:`{\sigma}_H` parameter in the original article, it's similar to the sigma in the coordinate space into :ocv:func:`bilateralFilter`.
    :param sigmaColor: :math:`{\sigma}_r` parameter in the original article, it's similar to the sigma in the color space into :ocv:func:`bilateralFilter`.
    :param mode: one form three modes ``DTF_NC``, ``DTF_RF`` and ``DTF_IC`` which corresponds to three modes for filtering 2D signals in the article.
    :param numIters: optional number of iterations used for filtering, 3 is quite enough.

For more details about Domain Transform filter parameters, see the original article [Gastal11]_ and `Domain Transform filter homepage <http://www.inf.ufrgs.br/~eslgastal/DomainTransform/>`_.

DTFilter::filter
------------------------------------
Produce domain transform filtering operation on source image.

.. ocv:function:: void DTFilter::filter(InputArray src, OutputArray dst, int dDepth = -1)

.. ocv:pyfunction:: cv2.DTFilter.filter(src, dst[, dDepth]) -> None

    :param src: filtering image with unsigned 8-bit or floating-point 32-bit depth and up to 4 channels.
    :param dst: destination image.
    :param dDepth: optional depth of the output image. ``dDepth`` can be set to -1, which will be equivalent to ``src.depth()``.
    
dtFilter
------------------------------------
Simple one-line Domain Transform filter call.
If you have multiple images to filter with the same guided image then use :ocv:class:`DTFilter` interface to avoid extra computations on initialization stage.

.. ocv:function:: void dtFilter(InputArray guide, InputArray src, OutputArray dst, double sigmaSpatial, double sigmaColor, int mode = DTF_NC, int numIters = 3)

.. ocv:pyfunction:: cv2.dtFilter(guide, src, sigmaSpatial, sigmaColor[, mode[, numIters]]) -> None

    :param guide: guided image (also called as joint image) with unsigned 8-bit or floating-point 32-bit depth and up to 4 channels.
    :param src: filtering image with unsigned 8-bit or floating-point 32-bit depth and up to 4 channels.
    :param sigmaSpatial: :math:`{\sigma}_H` parameter in the original article, it's similar to the sigma in the coordinate space into :ocv:func:`bilateralFilter`.
    :param sigmaColor: :math:`{\sigma}_r` parameter in the original article, it's similar to the sigma in the color space into :ocv:func:`bilateralFilter`.
    :param mode: one form three modes ``DTF_NC``, ``DTF_RF`` and ``DTF_IC`` which corresponds to three modes for filtering 2D signals in the article.
    :param numIters: optional number of iterations used for filtering, 3 is quite enough.
    
.. seealso:: :ocv:func:`bilateralFilter`, :ocv:func:`guidedFilter`, :ocv:func:`amFilter`

Guided Filter
====================================

This section describes interface for Guided Filter.
For more details about this filter see [Kaiming10]_ and References_.

GuidedFilter
------------------------------------
.. ocv:class:: GuidedFilter : public Algorithm

Interface for realizations of Guided Filter.

createGuidedFilter
------------------------------------
Factory method, create instance of :ocv:class:`GuidedFilter` and produce initialization routines.

.. ocv:function:: Ptr<GuidedFilter> createGuidedFilter(InputArray guide, int radius, double eps)

.. ocv:pyfunction:: cv2.createGuidedFilter(guide, radius, eps) -> instance

    :param guide: guided image (or array of images) with up to 3 channels, if it have more then 3 channels then only first 3 channels will be used.
    :param radius: radius of Guided Filter.
    :param eps: regularization term of Guided Filter. :math:`{eps}^2` is similar to the sigma in the color space into :ocv:func:`bilateralFilter`.

For more details about Guided Filter parameters, see the original article [Kaiming10]_.

GuidedFilter::filter
------------------------------------
Apply Guided Filter to the filtering image.

.. ocv:function:: void GuidedFilter::filter(InputArray src, OutputArray dst, int dDepth = -1)

.. ocv:pyfunction:: cv2.GuidedFilter.filter(src, dst[, dDepth]) -> None

    :param src: filtering image with any numbers of channels.
    :param dst: output image.
    :param dDepth: optional depth of the output image. ``dDepth`` can be set to -1, which will be equivalent to ``src.depth()``.
    
guidedFilter
------------------------------------
Simple one-line Guided Filter call.
If you have multiple images to filter with the same guided image then use :ocv:class:`GuidedFilter` interface to avoid extra computations on initialization stage.

.. ocv:function:: void guidedFilter(InputArray guide, InputArray src, OutputArray dst, int radius, double eps, int dDepth = -1)

.. ocv:pyfunction:: cv2.guidedFilter(guide, src, dst, radius, eps, [, dDepth]) -> None

    :param guide: guided image (or array of images) with up to 3 channels, if it have more then 3 channels then only first 3 channels will be used.
    :param src: filtering image with any numbers of channels.
    :param dst: output image.
    :param radius: radius of Guided Filter.
    :param eps: regularization term of Guided Filter. :math:`{eps}^2` is similar to the sigma in the color space into :ocv:func:`bilateralFilter`.
    :param dDepth: optional depth of the output image.
    
.. seealso:: :ocv:func:`bilateralFilter`, :ocv:func:`dtFilter`, :ocv:func:`amFilter`

Adaptive Manifold Filter
====================================

This section describes interface for Adaptive Manifold Filter.

For more details about this filter see [Gastal12]_ and References_.

AdaptiveManifoldFilter
------------------------------------
.. ocv:class:: AdaptiveManifoldFilter : public Algorithm
    
    Interface for Adaptive Manifold Filter realizations.
    
    Below listed optional parameters which may be set up with :ocv:func:`Algorithm::set` function.
    
    .. ocv:member:: double sigma_s = 16.0
    
        Spatial standard deviation.
        
    .. ocv:member:: double sigma_r = 0.2
    
        Color space standard deviation.
        
    .. ocv:member:: int tree_height = -1
    
        Height of the manifold tree (default = -1 : automatically computed).
    
    .. ocv:member:: int num_pca_iterations = 1
    
        Number of iterations to computed the eigenvector.
    
    .. ocv:member:: bool adjust_outliers = false
    
        Specify adjust outliers using Eq. 9 or not.
        
    .. ocv:member:: bool use_RNG = true
    
        Specify use random number generator to compute eigenvector or not.

createAMFilter
------------------------------------
Factory method, create instance of :ocv:class:`AdaptiveManifoldFilter` and produce some initialization routines.

.. ocv:function:: Ptr<AdaptiveManifoldFilter> createAMFilter(double sigma_s, double sigma_r, bool adjust_outliers = false)

.. ocv:pyfunction:: cv2.createAMFilter(sigma_s, sigma_r, adjust_outliers) -> instance

    :param sigma_s: spatial standard deviation.
    :param sigma_r: color space standard deviation, it is similar to the sigma in the color space into :ocv:func:`bilateralFilter`.
    :param adjust_outliers: optional, specify perform outliers adjust operation or not, (Eq. 9) in the original paper.

For more details about Adaptive Manifold Filter parameters, see the original article [Gastal12]_.

.. note::
    Joint images with `CV_8U` and `CV_16U` depth converted to images with `CV_32F` depth and [0; 1] color range before processing.
    Hence color space sigma `sigma_r` must be in [0; 1] range, unlike same sigmas in :ocv:func:`bilateralFilter` and :ocv:func:`dtFilter` functions.

AdaptiveManifoldFilter::filter
------------------------------------
Apply high-dimensional filtering using adaptive manifolds.

.. ocv:function:: void AdaptiveManifoldFilter::filter(InputArray src, OutputArray dst, InputArray joint = noArray())

.. ocv:pyfunction:: cv2.AdaptiveManifoldFilter.filter(src, dst[, joint]) -> None

    :param src: filtering image with any numbers of channels.
    :param dst: output image.
    :param joint: optional joint (also called as guided) image with any numbers of channels.
    
amFilter
------------------------------------
Simple one-line Adaptive Manifold Filter call.

.. ocv:function:: void amFilter(InputArray joint, InputArray src, OutputArray dst, double sigma_s, double sigma_r, bool adjust_outliers = false)

.. ocv:pyfunction:: cv2.amFilter(joint, src, dst, sigma_s, sigma_r, [, adjust_outliers]) -> None

    :param joint: joint (also  called as guided) image or array of images with any numbers of channels.
    :param src: filtering image with any numbers of channels.
    :param dst: output image.
    :param sigma_s: spatial standard deviation.
    :param sigma_r: color space standard deviation, it is similar to the sigma in the color space into :ocv:func:`bilateralFilter`.
    :param adjust_outliers: optional, specify perform outliers adjust operation or not, (Eq. 9) in the original paper.
    
.. note::
    Joint images with `CV_8U` and `CV_16U` depth converted to images with `CV_32F` depth and [0; 1] color range before processing.
    Hence color space sigma `sigma_r` must be in [0; 1] range, unlike same sigmas in :ocv:func:`bilateralFilter` and :ocv:func:`dtFilter` functions.
    
.. seealso:: :ocv:func:`bilateralFilter`, :ocv:func:`dtFilter`, :ocv:func:`guidedFilter`

Joint Bilateral Filter
====================================

jointBilateralFilter
------------------------------------
Applies the joint bilateral filter to an image.

.. ocv:function:: void jointBilateralFilter(InputArray joint, InputArray src, OutputArray dst, int d, double sigmaColor, double sigmaSpace, int borderType = BORDER_DEFAULT)

.. ocv:pyfunction:: cv2.jointBilateralFilter(joint, src, dst, d, sigmaColor, sigmaSpace, [, borderType]) -> None

    :param joint: Joint 8-bit or floating-point, 1-channel or 3-channel image.

    :param src: Source 8-bit or floating-point, 1-channel or 3-channel image with the same depth as joint image.

    :param dst: Destination image of the same size and type as  ``src`` .

    :param d: Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from  ``sigmaSpace`` .

    :param sigmaColor: Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see  ``sigmaSpace`` ) will be mixed together, resulting in larger areas of semi-equal color.

    :param sigmaSpace: Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see  ``sigmaColor`` ). When  ``d>0`` , it specifies the neighborhood size regardless of  ``sigmaSpace`` . Otherwise,  ``d``  is proportional to  ``sigmaSpace`` .

.. note:: :ocv:func:`bilateralFilter` and :ocv:func:`jointBilateralFilter` use L1 norm to compute difference between colors.
    
.. seealso:: :ocv:func:`bilateralFilter`, :ocv:func:`amFilter`

References
==========

..  [Gastal11] E. Gastal and M. Oliveira, "Domain Transform for Edge-Aware Image and Video Processing", Proceedings of SIGGRAPH, 2011, vol. 30, pp. 69:1 - 69:12.

    The paper is available `online <http://www.inf.ufrgs.br/~eslgastal/DomainTransform/>`__.


..  [Gastal12] E. Gastal and M. Oliveira, "Adaptive manifolds for real-time high-dimensional filtering," Proceedings of SIGGRAPH, 2012, vol. 31, pp. 33:1 - 33:13.

    The paper is available `online <http://inf.ufrgs.br/~eslgastal/AdaptiveManifolds/>`__.

    
..  [Kaiming10] Kaiming He et. al., "Guided Image Filtering," ECCV 2010, pp. 1 - 14.

    The paper is available `online <http://research.microsoft.com/en-us/um/people/kahe/eccv10/>`__.
    
    
.. [Tomasi98] Carlo Tomasi and Roberto Manduchi, “Bilateral filtering for gray and color images,” in Computer Vision, 1998. Sixth International Conference on . IEEE, 1998, pp. 839– 846.

    The paper is available `online <https://www.cs.duke.edu/~tomasi/papers/tomasi/tomasiIccv98.pdf>`__.
    

..  [Ziyang13] Ziyang Ma et al., "Constant Time Weighted Median Filtering for Stereo Matching and Beyond," ICCV, 2013, pp. 49 - 56.

    The paper is available `online <http://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Ma_Constant_Time_Weighted_2013_ICCV_paper.pdf>`__.
