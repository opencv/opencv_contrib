#pragma once
#include "opencv2/core.hpp"
#include <string>

namespace cv {
namespace hed {

/** @defgroup hed Holistically-nested Edge Detection
This module implements HED edge detection as described in:

Xie, S., & Tu, Z. (2015). Holistically-nested edge detection.
Proceedings of the IEEE international conference on computer vision.
https://arxiv.org/abs/1504.06375
*/

//! @addtogroup hed
//! @{

/** @brief HED edge detector using a pretrained Caffe model.

Unlike classical detectors such as Canny, HED uses a VGG-based
convolutional network trained on human-annotated boundaries to
produce semantically meaningful edge maps.

@note Requires hed_pretrained_bsds.caffemodel and deploy.prototxt.
Download from: https://vcl.ucsd.edu/hed/
*/
class CV_EXPORTS_W HEDDetector {
public:
    /** @brief Creates an HEDDetector instance.
    @param modelPath path to hed_pretrained_bsds.caffemodel
    @param protoPath path to deploy.prototxt
    */
    CV_WRAP static cv::Ptr<HEDDetector> create(
        const std::string& modelPath,
        const std::string& protoPath
    );

    /** @brief Detects edges in an image.
    @param image input BGR image (CV_8UC3)
    @return edge map as CV_32F with values in [0, 1]
    */
    CV_WRAP virtual cv::Mat detectEdges(cv::InputArray image) = 0;

    /** @brief Destructor. */
    virtual ~HEDDetector() {}
};

//! @}

} // namespace hed
} // namespace cv