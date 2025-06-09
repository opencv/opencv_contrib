// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CONTRIB_TRACKING_HPP
#define OPENCV_CONTRIB_TRACKING_HPP

#include "opencv2/core.hpp"
#include "opencv2/video/tracking.hpp"

namespace cv {
#ifndef CV_DOXYGEN
inline namespace tracking {
#endif

/** @defgroup tracking Tracking API
@{
    @defgroup tracking_detail Tracking API implementation details
    @defgroup tracking_legacy Legacy Tracking API
@}
*/

/** @addtogroup tracking
@{
Tracking is an important issue for many computer vision applications in real world scenario.
The development in this area is very fragmented and this API is an interface useful for plug several algorithms and compare them.
*/



/** @brief the KCF (Kernelized Correlation Filter) tracker

 * KCF is a novel tracking framework that utilizes properties of circulant matrix to enhance the processing speed.
 * This tracking method is an implementation of @cite KCF_ECCV which is extended to KCF with color-names features (@cite KCF_CN).
 * The original paper of KCF is available at <http://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf>
 * as well as the matlab implementation. For more information about KCF with color-names features, please refer to
 * <http://www.cvl.isy.liu.se/research/objrec/visualtracking/colvistrack/index.html>.
 */
class CV_EXPORTS_W TrackerKCF : public Tracker
{
protected:
    TrackerKCF();  // use ::create()
public:
    virtual ~TrackerKCF() CV_OVERRIDE;

    /**
    * \brief Feature type to be used in the tracking grayscale, colornames, compressed color-names
    * The modes available now:
    -   "GRAY" -- Use grayscale values as the feature
    -   "CN" -- Color-names feature
    */
    enum MODE {
      GRAY   = (1 << 0),
      CN     = (1 << 1),
      CUSTOM = (1 << 2)
    };

    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_WRAP Params();

        CV_PROP_RW float detect_thresh;         //!<  detection confidence threshold
        CV_PROP_RW float sigma;                 //!<  gaussian kernel bandwidth
        CV_PROP_RW float lambda;                //!<  regularization
        CV_PROP_RW float interp_factor;         //!<  linear interpolation factor for adaptation
        CV_PROP_RW float output_sigma_factor;   //!<  spatial bandwidth (proportional to target)
        CV_PROP_RW float pca_learning_rate;     //!<  compression learning rate
        CV_PROP_RW bool resize;                  //!<  activate the resize feature to improve the processing speed
        CV_PROP_RW bool split_coeff;             //!<  split the training coefficients into two matrices
        CV_PROP_RW bool wrap_kernel;             //!<  wrap around the kernel values
        CV_PROP_RW bool compress_feature;        //!<  activate the pca method to compress the features
        CV_PROP_RW int max_patch_size;           //!<  threshold for the ROI size
        CV_PROP_RW int compressed_size;          //!<  feature size after compression
        CV_PROP_RW int desc_pca;        //!<  compressed descriptors of TrackerKCF::MODE
        CV_PROP_RW int desc_npca;       //!<  non-compressed descriptors of TrackerKCF::MODE
    };

    /** @brief Create KCF tracker instance
    @param parameters KCF parameters TrackerKCF::Params
    */
    static CV_WRAP
    Ptr<TrackerKCF> create(const TrackerKCF::Params &parameters = TrackerKCF::Params());

    //void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    //bool update(InputArray image, CV_OUT Rect& boundingBox) CV_OVERRIDE;

    // FIXIT use interface
    typedef void (*FeatureExtractorCallbackFN)(const Mat, const Rect, Mat&);
    virtual void setFeatureExtractor(FeatureExtractorCallbackFN callback, bool pca_func = false) = 0;
};


//! @}

#ifndef CV_DOXYGEN
}
#endif
}  // namespace

#endif // OPENCV_CONTRIB_TRACKING_HPP
