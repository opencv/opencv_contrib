// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CONTRIB_TRACKING_HPP
#define OPENCV_CONTRIB_TRACKING_HPP

#include "opencv2/core.hpp"

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



/** @brief Base abstract class for the long-term tracker
 */
class CV_EXPORTS_W Tracker
{
protected:
    Tracker();
public:
    virtual ~Tracker();

    /** @brief Initialize the tracker with a known bounding box that surrounded the target
    @param image The initial frame
    @param boundingBox The initial bounding box
    */
    CV_WRAP virtual
    void init(InputArray image, const Rect& boundingBox) = 0;

    /** @brief Update the tracker, find the new most likely bounding box for the target
    @param image The current frame
    @param boundingBox The bounding box that represent the new target location, if true was returned, not
    modified otherwise

    @return True means that target was located and false means that tracker cannot locate target in
    current frame. Note, that latter *does not* imply that tracker has failed, maybe target is indeed
    missing from the frame (say, out of sight)
    */
    CV_WRAP virtual
    bool update(InputArray image, CV_OUT Rect& boundingBox) = 0;
};


/** @brief the CSRT tracker

The implementation is based on @cite Lukezic_IJCV2018 Discriminative Correlation Filter with Channel and Spatial Reliability
*/
class CV_EXPORTS_W TrackerCSRT : public Tracker
{
protected:
    TrackerCSRT();  // use ::create()
public:
    virtual ~TrackerCSRT() CV_OVERRIDE;

    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_WRAP Params();

        CV_PROP_RW bool use_hog;
        CV_PROP_RW bool use_color_names;
        CV_PROP_RW bool use_gray;
        CV_PROP_RW bool use_rgb;
        CV_PROP_RW bool use_channel_weights;
        CV_PROP_RW bool use_segmentation;

        CV_PROP_RW std::string window_function; //!<  Window function: "hann", "cheb", "kaiser"
        CV_PROP_RW float kaiser_alpha;
        CV_PROP_RW float cheb_attenuation;

        CV_PROP_RW float template_size;
        CV_PROP_RW float gsl_sigma;
        CV_PROP_RW float hog_orientations;
        CV_PROP_RW float hog_clip;
        CV_PROP_RW float padding;
        CV_PROP_RW float filter_lr;
        CV_PROP_RW float weights_lr;
        CV_PROP_RW int num_hog_channels_used;
        CV_PROP_RW int admm_iterations;
        CV_PROP_RW int histogram_bins;
        CV_PROP_RW float histogram_lr;
        CV_PROP_RW int background_ratio;
        CV_PROP_RW int number_of_scales;
        CV_PROP_RW float scale_sigma_factor;
        CV_PROP_RW float scale_model_max_area;
        CV_PROP_RW float scale_lr;
        CV_PROP_RW float scale_step;

        CV_PROP_RW float psr_threshold; //!< we lost the target, if the psr is lower than this.
    };

    /** @brief Create CSRT tracker instance
    @param parameters CSRT parameters TrackerCSRT::Params
    */
    static CV_WRAP
    Ptr<TrackerCSRT> create(const TrackerCSRT::Params &parameters = TrackerCSRT::Params());

    //void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    //bool update(InputArray image, CV_OUT Rect& boundingBox) CV_OVERRIDE;

    CV_WRAP virtual void setInitialMask(InputArray mask) = 0;
};



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



/** @brief The MIL algorithm trains a classifier in an online manner to separate the object from the
background.

Multiple Instance Learning avoids the drift problem for a robust tracking. The implementation is
based on @cite MIL .

Original code can be found here <http://vision.ucsd.edu/~bbabenko/project_miltrack.shtml>
 */
class CV_EXPORTS_W TrackerMIL : public Tracker
{
protected:
    TrackerMIL();  // use ::create()
public:
    virtual ~TrackerMIL() CV_OVERRIDE;

    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_WRAP Params();
        //parameters for sampler
        CV_PROP_RW float samplerInitInRadius;  //!< radius for gathering positive instances during init
        CV_PROP_RW int samplerInitMaxNegNum;  //!< # negative samples to use during init
        CV_PROP_RW float samplerSearchWinSize;  //!< size of search window
        CV_PROP_RW float samplerTrackInRadius;  //!< radius for gathering positive instances during tracking
        CV_PROP_RW int samplerTrackMaxPosNum;  //!< # positive samples to use during tracking
        CV_PROP_RW int samplerTrackMaxNegNum;  //!< # negative samples to use during tracking
        CV_PROP_RW int featureSetNumFeatures;  //!< # features
    };

    /** @brief Create MIL tracker instance
     *  @param parameters MIL parameters TrackerMIL::Params
     */
    static CV_WRAP
    Ptr<TrackerMIL> create(const TrackerMIL::Params &parameters = TrackerMIL::Params());


    //void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    //bool update(InputArray image, CV_OUT Rect& boundingBox) CV_OVERRIDE;
};



/** @brief the GOTURN (Generic Object Tracking Using Regression Networks) tracker
 *
 *  GOTURN (@cite GOTURN) is kind of trackers based on Convolutional Neural Networks (CNN). While taking all advantages of CNN trackers,
 *  GOTURN is much faster due to offline training without online fine-tuning nature.
 *  GOTURN tracker addresses the problem of single target tracking: given a bounding box label of an object in the first frame of the video,
 *  we track that object through the rest of the video. NOTE: Current method of GOTURN does not handle occlusions; however, it is fairly
 *  robust to viewpoint changes, lighting changes, and deformations.
 *  Inputs of GOTURN are two RGB patches representing Target and Search patches resized to 227x227.
 *  Outputs of GOTURN are predicted bounding box coordinates, relative to Search patch coordinate system, in format X1,Y1,X2,Y2.
 *  Original paper is here: <http://davheld.github.io/GOTURN/GOTURN.pdf>
 *  As long as original authors implementation: <https://github.com/davheld/GOTURN#train-the-tracker>
 *  Implementation of training algorithm is placed in separately here due to 3d-party dependencies:
 *  <https://github.com/Auron-X/GOTURN_Training_Toolkit>
 *  GOTURN architecture goturn.prototxt and trained model goturn.caffemodel are accessible on opencv_extra GitHub repository.
 */
class CV_EXPORTS_W TrackerGOTURN : public Tracker
{
protected:
    TrackerGOTURN();  // use ::create()
public:
    virtual ~TrackerGOTURN() CV_OVERRIDE;

    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_WRAP Params();
        CV_PROP_RW std::string modelTxt;
        CV_PROP_RW std::string modelBin;
    };

    /** @brief Constructor
    @param parameters GOTURN parameters TrackerGOTURN::Params
    */
    static CV_WRAP
    Ptr<TrackerGOTURN> create(const TrackerGOTURN::Params& parameters = TrackerGOTURN::Params());

    //void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    //bool update(InputArray image, CV_OUT Rect& boundingBox) CV_OVERRIDE;
};

//! @}

#ifndef CV_DOXYGEN
}
#endif
}  // namespace

#endif // OPENCV_CONTRIB_TRACKING_HPP
