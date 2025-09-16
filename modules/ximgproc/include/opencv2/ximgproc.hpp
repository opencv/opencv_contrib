// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_XIMGPROC_HPP__
#define __OPENCV_XIMGPROC_HPP__

#include "ximgproc/edge_filter.hpp"
#include "ximgproc/disparity_filter.hpp"
#include "ximgproc/sparse_match_interpolator.hpp"
#include "ximgproc/structured_edge_detection.hpp"
#include "ximgproc/edgeboxes.hpp"
#include "ximgproc/edge_drawing.hpp"
#include "ximgproc/scansegment.hpp"
#include "ximgproc/seeds.hpp"
#include "ximgproc/segmentation.hpp"
#include "ximgproc/fast_hough_transform.hpp"
#include "ximgproc/estimated_covariance.hpp"
#include "ximgproc/weighted_median_filter.hpp"
#include "ximgproc/slic.hpp"
#include "ximgproc/lsc.hpp"
#include "ximgproc/paillou_filter.hpp"
#include "ximgproc/fast_line_detector.hpp"
#include "ximgproc/deriche_filter.hpp"
#include "ximgproc/peilin.hpp"
#include "ximgproc/fourier_descriptors.hpp"
#include "ximgproc/ridgefilter.hpp"
#include "ximgproc/brightedges.hpp"
#include "ximgproc/run_length_morphology.hpp"
#include "ximgproc/edgepreserving_filter.hpp"
#include "ximgproc/color_match.hpp"
#include "ximgproc/radon_transform.hpp"
#include "ximgproc/find_ellipses.hpp"


/**
@defgroup ximgproc Extended Image Processing
@{
    @defgroup ximgproc_binarization Binarization

    @defgroup ximgproc_edge Structured forests for fast edge detection

    This module contains implementations of modern structured edge detection algorithms,
    i.e. algorithms which somehow takes into account pixel affinities in natural images.

    @defgroup ximgproc_edgeboxes EdgeBoxes

    @defgroup ximgproc_filters Filters

    @defgroup ximgproc_superpixel Superpixels

    @defgroup ximgproc_segmentation Image segmentation

    @defgroup ximgproc_fast_line_detector Fast line detector

    @defgroup ximgproc_edge_drawing Edge Drawing

    Edge Drawing (ED) algorithm for geometric feature extraction and validation.

    The Edge Drawing (ED) algorithm is a proactive approach to the edge detection problem.
    In contrast to many existing edge detection algorithms, which follow a subtractive
    approach (i.e., applying gradient filters and eliminating pixels based on several rules,
    such as non-maximal suppression and hysteresis in the Canny Edge Detector), the ED algorithm
    operates via an additive strategy. It selects edge pixels one by one and connects them,
    hence the name Edge Drawing.

    ED offers several key advantages:

    1. **Additive Strategy**: Instead of eliminating non-edge pixels after gradient filtering,
    ED incrementally builds up edge segments by selecting and connecting pixels based on
    their gradient response. This differs from traditional methods, which rely on
    non-maximal suppression and hysteresis to filter out non-edge pixels.

    2. **Edge Pixel Selection**: ED selects edge pixels by analyzing their local gradient
    response, while also considering neighboring pixels. This results in smoother and
    more continuous edge segments, as ED aims to maximize the overall gradient strength
    along the edge segment.

    3. **Edge Segment Formation**: Traditional methods, such as non-maximal suppression,
    check whether a pixel has the maximum gradient response along its gradient direction,
    eliminating it otherwise. However, this approach doesn't consider neighboring pixels,
    often resulting in lower-quality edge segments. ED, on the other hand, joins a set of
    edge pixels together by maximizing the total gradient response of the segment, leading
    to high-quality, well-localized edges.

    4. **Higher-Level Feature Extraction**: After forming edge segments, ED enables the
    extraction of higher-level geometric features such as lines, circles, ellipses, and
    other shapes, making it useful for tasks involving geometric feature extraction and validation.

    The ED algorithm produces continuous, smooth, and localized edge segments, making it ideal
    for applications requiring precise edge detection and geometric shape analysis.

    @defgroup ximgproc_fourier Fourier descriptors

    @defgroup ximgproc_run_length_morphology Binary morphology on run-length encoded image

    These functions support morphological operations on binary images. In order to be fast and space efficient binary images are encoded with a run-length representation.
    This representation groups continuous horizontal sequences of "on" pixels together in a "run". A run is charactarized by the column position of the first pixel in the run, the column
    position of the last pixel in the run and the row position. This representation is very compact for binary images which contain large continuous areas of "on" and "off" pixels. A checkerboard
    pattern would be a good example. The representation is not so suitable for binary images created from random noise images or other images where little correlation between neighboring pixels
    exists.

    The morphological operations supported here are very similar to the operations supported in the imgproc module. In general they are fast. However on several occasions they are slower than the functions
    from imgproc. The structuring elements of cv::MORPH_RECT and cv::MORPH_CROSS have very good support from the imgproc module. Also small structuring elements are very fast in imgproc (presumably
    due to opencl support). Therefore the functions from this module are recommended for larger structuring elements (cv::MORPH_ELLIPSE or self defined structuring elements). A sample application
    (run_length_morphology_demo) is supplied which allows to compare the speed of some morphological operations for the functions using run-length encoding and the imgproc functions for a given image.

    Run length encoded images are stored in standard opencv images. Images have a single column of cv::Point3i elements. The number of rows is the number of run + 1. The first row contains
    the size of the original (not encoded) image.  For the runs the following mapping is used (x: column begin, y: column end (last column), z: row).

    The size of the original image is required for compatibility with the imgproc functions when the boundary handling requires that pixel outside the image boundary are
    "on".
@}
*/

namespace cv
{
namespace ximgproc
{

//! @addtogroup ximgproc_binarization
//! @{

enum ThinningTypes{
    THINNING_ZHANGSUEN    = 0, // Thinning technique of Zhang-Suen
    THINNING_GUOHALL      = 1  // Thinning technique of Guo-Hall
};

/**
* @brief Specifies the binarization method to use in cv::ximgproc::niBlackThreshold
*/
enum LocalBinarizationMethods{
	BINARIZATION_NIBLACK = 0, //!< Classic Niblack binarization. See @cite Niblack1985 .
	BINARIZATION_SAUVOLA = 1, //!< Sauvola's technique. See @cite Sauvola1997 .
	BINARIZATION_WOLF = 2,    //!< Wolf's technique. See @cite Wolf2004 .
	BINARIZATION_NICK = 3     //!< NICK technique. See @cite Khurshid2009 .
};

/** @brief Performs thresholding on input images using Niblack's technique or some of the
popular variations it inspired.

The function transforms a grayscale image to a binary image according to the formulae:
-   **THRESH_BINARY**
    \f[dst(x,y) =  \fork{\texttt{maxValue}}{if \(src(x,y) > T(x,y)\)}{0}{otherwise}\f]
-   **THRESH_BINARY_INV**
    \f[dst(x,y) =  \fork{0}{if \(src(x,y) > T(x,y)\)}{\texttt{maxValue}}{otherwise}\f]
where \f$T(x,y)\f$ is a threshold calculated individually for each pixel.

The threshold value \f$T(x, y)\f$ is determined based on the binarization method chosen. For
classic Niblack, it is the mean minus \f$ k \f$ times standard deviation of
\f$\texttt{blockSize} \times\texttt{blockSize}\f$ neighborhood of \f$(x, y)\f$.

The function can't process the image in-place.

@param _src Source 8-bit single-channel image.
@param _dst Destination image of the same size and the same type as src.
@param maxValue Non-zero value assigned to the pixels for which the condition is satisfied,
used with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
@param type Thresholding type, see cv::ThresholdTypes.
@param blockSize Size of a pixel neighborhood that is used to calculate a threshold value
for the pixel: 3, 5, 7, and so on.
@param k The user-adjustable parameter used by Niblack and inspired techniques. For Niblack, this is
normally a value between 0 and 1 that is multiplied with the standard deviation and subtracted from
the mean.
@param binarizationMethod Binarization method to use. By default, Niblack's technique is used.
Other techniques can be specified, see cv::ximgproc::LocalBinarizationMethods.
@param r The user-adjustable parameter used by Sauvola's technique. This is the dynamic range
of standard deviation.
@sa  threshold, adaptiveThreshold
 */
CV_EXPORTS_W void niBlackThreshold( InputArray _src, OutputArray _dst,
                                    double maxValue, int type,
                                    int blockSize, double k, int binarizationMethod = BINARIZATION_NIBLACK,
                                    double r = 128 );

/** @brief Performs binary image thinning to obtain a skeletonized representation of the input image.

This function applies a thinning algorithm, reducing the binary blobs in the input image to a skeletal form.
By default, it uses the Zhang-Suen technique, which iteratively removes pixels from the boundaries of the blobs
while preserving the overall structure and connectivity of the objects.

@param src Source image: an 8-bit, single-channel binary image where the blobs are represented by pixels with a value of 255 (white),
and the background is 0 (black).
@param dst Destination image of the same size and type as src, where the result of the thinning operation will be stored.
This operation can be performed in-place, meaning `src` and `dst` can be the same.
@param thinningType The thinning algorithm to apply. By default, the Zhang-Suen algorithm is used. See cv::ximgproc::ThinningTypes for other options.
*/
CV_EXPORTS_W void thinning(InputArray src, OutputArray dst, int thinningType = THINNING_ZHANGSUEN);


/** @brief Performs anisotropic diffusion on an image.

 The function applies Perona-Malik anisotropic diffusion to an image. This is the solution to the partial differential equation:

 \f[{\frac  {\partial I}{\partial t}}={\mathrm  {div}}\left(c(x,y,t)\nabla I\right)=\nabla c\cdot \nabla I+c(x,y,t)\Delta I\f]

 Suggested functions for c(x,y,t) are:

 \f[c\left(\|\nabla I\|\right)=e^{{-\left(\|\nabla I\|/K\right)^{2}}}\f]

 or

 \f[ c\left(\|\nabla I\|\right)={\frac {1}{1+\left({\frac  {\|\nabla I\|}{K}}\right)^{2}}} \f]

 @param src Source image with 3 channels.
 @param dst Destination image of the same size and the same number of channels as src .
 @param alpha The amount of time to step forward by on each iteration (normally, it's between 0 and 1).
 @param K sensitivity to the edges
 @param niters The number of iterations
*/
CV_EXPORTS_W void anisotropicDiffusion(InputArray src, OutputArray dst, float alpha, float K, int niters );

//! @}

}
}

#endif // __OPENCV_XIMGPROC_HPP__
