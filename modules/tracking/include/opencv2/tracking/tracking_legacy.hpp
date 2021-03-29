/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#ifndef OPENCV_TRACKING_LEGACY_HPP
#define OPENCV_TRACKING_LEGACY_HPP

/*
 * Partially based on:
 * ====================================================================================================================
 *  - [AAM] S. Salti, A. Cavallaro, L. Di Stefano, Adaptive Appearance Modeling for Video Tracking: Survey and Evaluation
 *  - [AMVOT] X. Li, W. Hu, C. Shen, Z. Zhang, A. Dick, A. van den Hengel, A Survey of Appearance Models in Visual Object Tracking
 *
 * This Tracking API has been designed with PlantUML. If you modify this API please change UML files under modules/tracking/doc/uml
 *
 */

#include "tracking_internals.hpp"

namespace cv {
namespace legacy {
#ifndef CV_DOXYGEN
inline namespace tracking {
#endif
using namespace cv::detail::tracking;

/** @addtogroup tracking_legacy
@{
*/

/************************************ Tracker Base Class ************************************/

/** @brief Base abstract class for the long-term tracker:
 */
class CV_EXPORTS_W Tracker : public virtual Algorithm
{
 public:
  Tracker();
  virtual ~Tracker() CV_OVERRIDE;

  /** @brief Initialize the tracker with a known bounding box that surrounded the target
    @param image The initial frame
    @param boundingBox The initial bounding box

    @return True if initialization went succesfully, false otherwise
     */
  CV_WRAP bool init( InputArray image, const Rect2d& boundingBox );

  /** @brief Update the tracker, find the new most likely bounding box for the target
    @param image The current frame
    @param boundingBox The bounding box that represent the new target location, if true was returned, not
    modified otherwise

    @return True means that target was located and false means that tracker cannot locate target in
    current frame. Note, that latter *does not* imply that tracker has failed, maybe target is indeed
    missing from the frame (say, out of sight)
     */
  CV_WRAP bool update( InputArray image, CV_OUT Rect2d& boundingBox );

  virtual void read( const FileNode& fn ) CV_OVERRIDE = 0;
  virtual void write( FileStorage& fs ) const CV_OVERRIDE = 0;

 protected:

  virtual bool initImpl( const Mat& image, const Rect2d& boundingBox ) = 0;
  virtual bool updateImpl( const Mat& image, Rect2d& boundingBox ) = 0;

  bool isInit;

  Ptr<TrackerContribFeatureSet> featureSet;
  Ptr<TrackerContribSampler> sampler;
  Ptr<TrackerModel> model;
};


/************************************ Specific Tracker Classes ************************************/

/** @brief The MIL algorithm trains a classifier in an online manner to separate the object from the
background.

Multiple Instance Learning avoids the drift problem for a robust tracking. The implementation is
based on @cite MIL .

Original code can be found here <http://vision.ucsd.edu/~bbabenko/project_miltrack.shtml>
 */
class CV_EXPORTS_W TrackerMIL : public cv::legacy::Tracker
{
 public:
  struct CV_EXPORTS Params : cv::TrackerMIL::Params
  {
    void read( const FileNode& fn );
    void write( FileStorage& fs ) const;
  };

  /** @brief Constructor
    @param parameters MIL parameters TrackerMIL::Params
     */
  static Ptr<legacy::TrackerMIL> create(const TrackerMIL::Params &parameters);

  CV_WRAP static Ptr<legacy::TrackerMIL> create();

  virtual ~TrackerMIL() CV_OVERRIDE {}
};

/** @brief the Boosting tracker

This is a real-time object tracking based on a novel on-line version of the AdaBoost algorithm.
The classifier uses the surrounding background as negative examples in update step to avoid the
drifting problem. The implementation is based on @cite OLB .
 */
class CV_EXPORTS_W TrackerBoosting : public cv::legacy::Tracker
{
 public:
  struct CV_EXPORTS Params
  {
    Params();
    int numClassifiers;  //!<the number of classifiers to use in a OnlineBoosting algorithm
    float samplerOverlap;  //!<search region parameters to use in a OnlineBoosting algorithm
    float samplerSearchFactor;  //!< search region parameters to use in a OnlineBoosting algorithm
    int iterationInit;  //!<the initial iterations
    int featureSetNumFeatures;  //!< # features
    /**
     * \brief Read parameters from a file
     */
    void read( const FileNode& fn );

    /**
     * \brief Write parameters to a file
     */
    void write( FileStorage& fs ) const;
  };

  /** @brief Constructor
    @param parameters BOOSTING parameters TrackerBoosting::Params
     */
  static Ptr<legacy::TrackerBoosting> create(const TrackerBoosting::Params &parameters);

  CV_WRAP static Ptr<legacy::TrackerBoosting> create();

  virtual ~TrackerBoosting() CV_OVERRIDE {}
};

/** @brief the Median Flow tracker

Implementation of a paper @cite MedianFlow .

The tracker is suitable for very smooth and predictable movements when object is visible throughout
the whole sequence. It's quite and accurate for this type of problems (in particular, it was shown
by authors to outperform MIL). During the implementation period the code at
<http://www.aonsquared.co.uk/node/5>, the courtesy of the author Arthur Amarra, was used for the
reference purpose.
 */
class CV_EXPORTS_W TrackerMedianFlow : public cv::legacy::Tracker
{
 public:
  struct CV_EXPORTS Params
  {
    Params(); //!<default constructor
              //!<note that the default values of parameters are recommended for most of use cases
    int pointsInGrid;      //!<square root of number of keypoints used; increase it to trade
                           //!<accurateness for speed
    cv::Size winSize;      //!<window size parameter for Lucas-Kanade optical flow
    int maxLevel;          //!<maximal pyramid level number for Lucas-Kanade optical flow
    TermCriteria termCriteria; //!<termination criteria for Lucas-Kanade optical flow
    cv::Size winSizeNCC;   //!<window size around a point for normalized cross-correlation check
    double maxMedianLengthOfDisplacementDifference; //!<criterion for loosing the tracked object

    void read( const FileNode& /*fn*/ );
    void write( FileStorage& /*fs*/ ) const;
  };

  /** @brief Constructor
    @param parameters Median Flow parameters TrackerMedianFlow::Params
    */
  static Ptr<legacy::TrackerMedianFlow> create(const TrackerMedianFlow::Params &parameters);

  CV_WRAP static Ptr<legacy::TrackerMedianFlow> create();

  virtual ~TrackerMedianFlow() CV_OVERRIDE {}
};

/** @brief the TLD (Tracking, learning and detection) tracker

TLD is a novel tracking framework that explicitly decomposes the long-term tracking task into
tracking, learning and detection.

The tracker follows the object from frame to frame. The detector localizes all appearances that
have been observed so far and corrects the tracker if necessary. The learning estimates detector's
errors and updates it to avoid these errors in the future. The implementation is based on @cite TLD .

The Median Flow algorithm (see cv::TrackerMedianFlow) was chosen as a tracking component in this
implementation, following authors. The tracker is supposed to be able to handle rapid motions, partial
occlusions, object absence etc.
 */
class CV_EXPORTS_W TrackerTLD : public cv::legacy::Tracker
{
 public:
  struct CV_EXPORTS Params
  {
    Params();
    void read( const FileNode& /*fn*/ );
    void write( FileStorage& /*fs*/ ) const;
  };

  /** @brief Constructor
    @param parameters TLD parameters TrackerTLD::Params
     */
  static Ptr<legacy::TrackerTLD> create(const TrackerTLD::Params &parameters);

  CV_WRAP static Ptr<legacy::TrackerTLD> create();

  virtual ~TrackerTLD() CV_OVERRIDE {}
};

/** @brief the KCF (Kernelized Correlation Filter) tracker

 * KCF is a novel tracking framework that utilizes properties of circulant matrix to enhance the processing speed.
 * This tracking method is an implementation of @cite KCF_ECCV which is extended to KCF with color-names features (@cite KCF_CN).
 * The original paper of KCF is available at <http://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf>
 * as well as the matlab implementation. For more information about KCF with color-names features, please refer to
 * <http://www.cvl.isy.liu.se/research/objrec/visualtracking/colvistrack/index.html>.
 */
class CV_EXPORTS_W TrackerKCF : public cv::legacy::Tracker
{
public:
  /**
  * \brief Feature type to be used in the tracking grayscale, colornames, compressed color-names
  * The modes available now:
  -   "GRAY" -- Use grayscale values as the feature
  -   "CN" -- Color-names feature
  */
  typedef enum cv::tracking::TrackerKCF::MODE MODE;

  struct CV_EXPORTS Params : cv::tracking::TrackerKCF::Params
  {
    void read(const FileNode& /*fn*/);
    void write(FileStorage& /*fs*/) const;
  };

  virtual void setFeatureExtractor(void(*)(const Mat, const Rect, Mat&), bool pca_func = false) = 0;

  /** @brief Constructor
  @param parameters KCF parameters TrackerKCF::Params
  */
  static Ptr<legacy::TrackerKCF> create(const TrackerKCF::Params &parameters);

  CV_WRAP static Ptr<legacy::TrackerKCF> create();

  virtual ~TrackerKCF() CV_OVERRIDE {}
};

#if 0  // legacy variant is not available
/** @brief the GOTURN (Generic Object Tracking Using Regression Networks) tracker

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
class CV_EXPORTS_W TrackerGOTURN : public cv::legacy::Tracker
{
public:
  struct CV_EXPORTS Params
  {
    Params();
    void read(const FileNode& /*fn*/);
    void write(FileStorage& /*fs*/) const;
    String modelTxt;
    String modelBin;
  };

  /** @brief Constructor
  @param parameters GOTURN parameters TrackerGOTURN::Params
  */
  static Ptr<legacy::TrackerGOTURN> create(const TrackerGOTURN::Params &parameters);

  CV_WRAP static Ptr<legacy::TrackerGOTURN> create();

  virtual ~TrackerGOTURN() CV_OVERRIDE {}
};
#endif

/** @brief the MOSSE (Minimum Output Sum of Squared %Error) tracker

The implementation is based on @cite MOSSE Visual Object Tracking using Adaptive Correlation Filters
@note this tracker works with grayscale images, if passed bgr ones, they will get converted internally.
*/

class CV_EXPORTS_W TrackerMOSSE : public cv::legacy::Tracker
{
 public:
  /** @brief Constructor
  */
  CV_WRAP static Ptr<legacy::TrackerMOSSE> create();

  virtual ~TrackerMOSSE() CV_OVERRIDE {}
};


/************************************ MultiTracker Class ---By Laksono Kurnianggoro---) ************************************/
/** @brief This class is used to track multiple objects using the specified tracker algorithm.

* The %MultiTracker is naive implementation of multiple object tracking.
* It process the tracked objects independently without any optimization accross the tracked objects.
*/
class CV_EXPORTS_W MultiTracker : public Algorithm
{
public:

  /**
  * \brief Constructor.
  */
  CV_WRAP MultiTracker();

  /**
  * \brief Destructor
  */
  ~MultiTracker() CV_OVERRIDE;

  /**
  * \brief Add a new object to be tracked.
  *
  * @param newTracker tracking algorithm to be used
  * @param image input image
  * @param boundingBox a rectangle represents ROI of the tracked object
  */
  CV_WRAP bool add(Ptr<cv::legacy::Tracker> newTracker, InputArray image, const Rect2d& boundingBox);

  /**
  * \brief Add a set of objects to be tracked.
  * @param newTrackers list of tracking algorithms to be used
  * @param image input image
  * @param boundingBox list of the tracked objects
  */
  bool add(std::vector<Ptr<legacy::Tracker> > newTrackers, InputArray image, std::vector<Rect2d> boundingBox);

  /**
  * \brief Update the current tracking status.
  * The result will be saved in the internal storage.
  * @param image input image
  */
  bool update(InputArray image);

  /**
  * \brief Update the current tracking status.
  * @param image input image
  * @param boundingBox the tracking result, represent a list of ROIs of the tracked objects.
  */
  CV_WRAP bool update(InputArray image, CV_OUT std::vector<Rect2d> & boundingBox);

  /**
  * \brief Returns a reference to a storage for the tracked objects, each object corresponds to one tracker algorithm
  */
  CV_WRAP const std::vector<Rect2d>& getObjects() const;

  /**
  * \brief Returns a pointer to a new instance of MultiTracker
  */
  CV_WRAP static Ptr<MultiTracker> create();

protected:
  //!<  storage for the tracker algorithms.
  std::vector< Ptr<Tracker> > trackerList;

  //!<  storage for the tracked objects, each object corresponds to one tracker algorithm.
  std::vector<Rect2d> objects;
};

/************************************ Multi-Tracker Classes ---By Tyan Vladimir---************************************/

/** @brief Base abstract class for the long-term Multi Object Trackers:

@sa Tracker, MultiTrackerTLD
*/
class CV_EXPORTS MultiTracker_Alt
{
public:
  /** @brief Constructor for Multitracker
  */
  MultiTracker_Alt()
  {
    targetNum = 0;
  }

  /** @brief Add a new target to a tracking-list and initialize the tracker with a known bounding box that surrounded the target
  @param image The initial frame
  @param boundingBox The initial bounding box of target
  @param tracker_algorithm Multi-tracker algorithm

  @return True if new target initialization went succesfully, false otherwise
  */
  bool addTarget(InputArray image, const Rect2d& boundingBox, Ptr<legacy::Tracker> tracker_algorithm);

  /** @brief Update all trackers from the tracking-list, find a new most likely bounding boxes for the targets
  @param image The current frame

  @return True means that all targets were located and false means that tracker couldn't locate one of the targets in
  current frame. Note, that latter *does not* imply that tracker has failed, maybe target is indeed
  missing from the frame (say, out of sight)
  */
  bool update(InputArray image);

  /** @brief Current number of targets in tracking-list
  */
  int targetNum;

  /** @brief Trackers list for Multi-Object-Tracker
  */
  std::vector <Ptr<Tracker> > trackers;

  /** @brief Bounding Boxes list for Multi-Object-Tracker
  */
  std::vector <Rect2d> boundingBoxes;
  /** @brief List of randomly generated colors for bounding boxes display
  */
  std::vector<Scalar> colors;
};

/** @brief Multi Object %Tracker for TLD.

TLD is a novel tracking framework that explicitly decomposes
the long-term tracking task into tracking, learning and detection.

The tracker follows the object from frame to frame. The detector localizes all appearances that
have been observed so far and corrects the tracker if necessary. The learning estimates detector's
errors and updates it to avoid these errors in the future. The implementation is based on @cite TLD .

The Median Flow algorithm (see cv::TrackerMedianFlow) was chosen as a tracking component in this
implementation, following authors. The tracker is supposed to be able to handle rapid motions, partial
occlusions, object absence etc.

@sa Tracker, MultiTracker, TrackerTLD
*/
class CV_EXPORTS MultiTrackerTLD : public MultiTracker_Alt
{
public:
  /** @brief Update all trackers from the tracking-list, find a new most likely bounding boxes for the targets by
  optimized update method using some techniques to speedup calculations specifically for MO TLD. The only limitation
  is that all target bounding boxes should have approximately same aspect ratios. Speed boost is around 20%

  @param image The current frame.

  @return True means that all targets were located and false means that tracker couldn't locate one of the targets in
  current frame. Note, that latter *does not* imply that tracker has failed, maybe target is indeed
  missing from the frame (say, out of sight)
  */
  bool update_opt(InputArray image);
};

/*********************************** CSRT ************************************/
/** @brief the CSRT tracker

The implementation is based on @cite Lukezic_IJCV2018 Discriminative Correlation Filter with Channel and Spatial Reliability
*/
class CV_EXPORTS_W TrackerCSRT : public cv::legacy::Tracker
{
public:
  struct CV_EXPORTS Params : cv::tracking::TrackerCSRT::Params
  {
    /**
    * \brief Read parameters from a file
    */
    void read(const FileNode& /*fn*/);

    /**
    * \brief Write parameters to a file
    */
    void write(cv::FileStorage& fs) const;
  };

  /** @brief Constructor
  @param parameters CSRT parameters TrackerCSRT::Params
  */
  static Ptr<legacy::TrackerCSRT> create(const TrackerCSRT::Params &parameters);

  CV_WRAP static Ptr<legacy::TrackerCSRT> create();

  CV_WRAP virtual void setInitialMask(InputArray mask) = 0;

  virtual ~TrackerCSRT() CV_OVERRIDE {}
};


CV_EXPORTS_W Ptr<cv::Tracker> upgradeTrackingAPI(const Ptr<legacy::Tracker>& legacy_tracker);

//! @}

#ifndef CV_DOXYGEN
}  // namespace
#endif
}}  // namespace

#endif // OPENCV_TRACKING_LEGACY_HPP
