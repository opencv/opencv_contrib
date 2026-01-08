// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021, Dr Seng Cheong Loke (lokesengcheong@gmail.com)

#ifndef __OPENCV_XIMGPROC_SCANSEGMENT_HPP__
#define __OPENCV_XIMGPROC_SCANSEGMENT_HPP__

#include <opencv2/core.hpp>

namespace cv { namespace ximgproc {

/** @brief Class implementing the F-DBSCAN (Accelerated superpixel image segmentation with a parallelized DBSCAN algorithm) superpixels
algorithm by Loke SC, et al. @cite loke2021accelerated for original paper.

The algorithm uses a parallelised DBSCAN cluster search that is resistant to noise, competitive in segmentation quality, and faster than
existing superpixel segmentation methods. When tested on the Berkeley Segmentation Dataset, the average processing speed is 175 frames/s
with a Boundary Recall of 0.797 and an Achievable Segmentation Accuracy of 0.944. The computational complexity is quadratic O(n2) and
more suited to smaller images, but can still process a 2MP colour image faster than the SEEDS algorithm in OpenCV. The output is deterministic
when the number of processing threads is fixed, and requires the source image to be in Lab colour format.
*/
class CV_EXPORTS_W ScanSegment : public Algorithm
{
public:
    virtual ~ScanSegment();

    /** @brief Returns the actual superpixel segmentation from the last image processed using iterate.

    Returns zero if no image has been processed.
    */
    CV_WRAP virtual int getNumberOfSuperpixels() = 0;

    /** @brief Calculates the superpixel segmentation on a given image with the initialized
    parameters in the ScanSegment object.

    This function can be called again for other images without the need of initializing the algorithm with createScanSegment().
    This save the computational cost of allocating memory for all the structures of the algorithm.

    @param img Input image. Supported format: CV_8UC3. Image size must match with the initialized
    image size with the function createScanSegment(). It MUST be in Lab color space.
    */
    CV_WRAP virtual void iterate(InputArray img) = 0;

    /** @brief Returns the segmentation labeling of the image.

    Each label represents a superpixel, and each pixel is assigned to one superpixel label.

    @param labels_out Return: A CV_32UC1 integer array containing the labels of the superpixel
    segmentation. The labels are in the range [0, getNumberOfSuperpixels()].
    */
    CV_WRAP virtual void getLabels(OutputArray labels_out) = 0;

    /** @brief Returns the mask of the superpixel segmentation stored in the ScanSegment object.

    The function return the boundaries of the superpixel segmentation.

    @param image Return: CV_8UC1 image mask where -1 indicates that the pixel is a superpixel border, and 0 otherwise.
    @param thick_line If false, the border is only one pixel wide, otherwise all pixels at the border are masked.
    */
    CV_WRAP virtual void getLabelContourMask(OutputArray image, bool thick_line = false) = 0;
};

/** @brief Initializes a ScanSegment object.

The function initializes a ScanSegment object for the input image. It stores the parameters of
the image: image_width and image_height. It also sets the parameters of the F-DBSCAN superpixel
algorithm, which are: num_superpixels, threads, and merge_small.

@param image_width Image width.
@param image_height Image height.
@param num_superpixels Desired number of superpixels. Note that the actual number may be smaller
due to restrictions (depending on the image size). Use getNumberOfSuperpixels() to
get the actual number.
@param slices Number of processing threads for parallelisation. Setting -1 uses the maximum number
of threads. In practice, four threads is enough for smaller images and eight threads for larger ones.
@param merge_small merge small segments to give the desired number of superpixels. Processing is
much faster without merging, but many small segments will be left in the image.
*/
CV_EXPORTS_W cv::Ptr<ScanSegment> createScanSegment(int image_width, int image_height, int num_superpixels, int slices = 8, bool merge_small = true);

}}  // namespace
#endif
