// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_TRACKING_DETAIL_HPP
#define OPENCV_TRACKING_DETAIL_HPP

/*
 * Partially based on:
 * ====================================================================================================================
 *  - [AAM] S. Salti, A. Cavallaro, L. Di Stefano, Adaptive Appearance Modeling for Video Tracking: Survey and Evaluation
 *  - [AMVOT] X. Li, W. Hu, C. Shen, Z. Zhang, A. Dick, A. van den Hengel, A Survey of Appearance Models in Visual Object Tracking
 *
 * This Tracking API has been designed with PlantUML. If you modify this API please change UML files under modules/tracking/doc/uml
 *
 */

#include "opencv2/video/detail/tracking.private.hpp"

#include "feature.hpp"  // CvHaarEvaluator
#include "onlineBoosting.hpp"  // StrongClassifierDirectSelection

namespace cv {
namespace detail {
inline namespace tracking {

/** @addtogroup tracking_detail
@{

Long-term optical tracking API
------------------------------

Long-term optical tracking is an important issue for many computer vision applications in
real world scenario. The development in this area is very fragmented and this API is an unique
interface useful for plug several algorithms and compare them. This work is partially based on
@cite AAM and @cite AMVOT .

These algorithms start from a bounding box of the target and with their internal representation they
avoid the drift during the tracking. These long-term trackers are able to evaluate online the
quality of the location of the target in the new frame, without ground truth.

There are three main components: the TrackerContribSampler, the TrackerContribFeatureSet and the TrackerModel. The
first component is the object that computes the patches over the frame based on the last target
location. The TrackerContribFeatureSet is the class that manages the Features, is possible plug many kind
of these (HAAR, HOG, LBP, Feature2D, etc). The last component is the internal representation of the
target, it is the appearance model. It stores all state candidates and compute the trajectory (the
most likely target states). The class TrackerTargetState represents a possible state of the target.
The TrackerContribSampler and the TrackerContribFeatureSet are the visual representation of the target, instead
the TrackerModel is the statistical model.

A recent benchmark between these algorithms can be found in @cite OOT

Creating Your Own %Tracker
--------------------

If you want to create a new tracker, here's what you have to do. First, decide on the name of the class
for the tracker (to meet the existing style, we suggest something with prefix "tracker", e.g.
trackerMIL, trackerBoosting) -- we shall refer to this choice as to "classname" in subsequent.

-   Declare your tracker in modules/tracking/include/opencv2/tracking/tracker.hpp. Your tracker should inherit from
    Tracker (please, see the example below). You should declare the specialized Param structure,
    where you probably will want to put the data, needed to initialize your tracker. You should
    get something similar to :
@code
        class CV_EXPORTS_W TrackerMIL : public Tracker
        {
         public:
          struct CV_EXPORTS Params
          {
            Params();
            //parameters for sampler
            float samplerInitInRadius;  // radius for gathering positive instances during init
            int samplerInitMaxNegNum;  // # negative samples to use during init
            float samplerSearchWinSize;  // size of search window
            float samplerTrackInRadius;  // radius for gathering positive instances during tracking
            int samplerTrackMaxPosNum;  // # positive samples to use during tracking
            int samplerTrackMaxNegNum;  // # negative samples to use during tracking
            int featureSetNumFeatures;  // #features

            void read( const FileNode& fn );
            void write( FileStorage& fs ) const;
          };
@endcode
    of course, you can also add any additional methods of your choice. It should be pointed out,
    however, that it is not expected to have a constructor declared, as creation should be done via
    the corresponding create() method.
-   Finally, you should implement the function with signature :
@code
        Ptr<classname> classname::create(const classname::Params &parameters){
            ...
        }
@endcode
    That function can (and probably will) return a pointer to some derived class of "classname",
    which will probably have a real constructor.

Every tracker has three component TrackerContribSampler, TrackerContribFeatureSet and TrackerModel. The first two
are instantiated from Tracker base class, instead the last component is abstract, so you must
implement your TrackerModel.

### TrackerContribSampler

TrackerContribSampler is already instantiated, but you should define the sampling algorithm and add the
classes (or single class) to TrackerContribSampler. You can choose one of the ready implementation as
TrackerContribSamplerCSC or you can implement your sampling method, in this case the class must inherit
TrackerContribSamplerAlgorithm. Fill the samplingImpl method that writes the result in "sample" output
argument.

Example of creating specialized TrackerContribSamplerAlgorithm TrackerContribSamplerCSC : :
@code
    class CV_EXPORTS_W TrackerContribSamplerCSC : public TrackerContribSamplerAlgorithm
    {
     public:
      TrackerContribSamplerCSC( const TrackerContribSamplerCSC::Params &parameters = TrackerContribSamplerCSC::Params() );
      ~TrackerContribSamplerCSC();
      ...

     protected:
      bool samplingImpl( const Mat& image, Rect boundingBox, std::vector<Mat>& sample );
      ...

    };
@endcode

Example of adding TrackerContribSamplerAlgorithm to TrackerContribSampler : :
@code
    //sampler is the TrackerContribSampler
    Ptr<TrackerContribSamplerAlgorithm> CSCSampler = new TrackerContribSamplerCSC( CSCparameters );
    if( !sampler->addTrackerSamplerAlgorithm( CSCSampler ) )
     return false;

    //or add CSC sampler with default parameters
    //sampler->addTrackerSamplerAlgorithm( "CSC" );
@endcode
@sa
   TrackerContribSamplerCSC, TrackerContribSamplerAlgorithm

### TrackerContribFeatureSet

TrackerContribFeatureSet is already instantiated (as first) , but you should define what kinds of features
you'll use in your tracker. You can use multiple feature types, so you can add a ready
implementation as TrackerContribFeatureHAAR in your TrackerContribFeatureSet or develop your own implementation.
In this case, in the computeImpl method put the code that extract the features and in the selection
method optionally put the code for the refinement and selection of the features.

Example of creating specialized TrackerFeature TrackerContribFeatureHAAR : :
@code
    class CV_EXPORTS_W TrackerContribFeatureHAAR : public TrackerFeature
    {
     public:
      TrackerContribFeatureHAAR( const TrackerContribFeatureHAAR::Params &parameters = TrackerContribFeatureHAAR::Params() );
      ~TrackerContribFeatureHAAR();
      void selection( Mat& response, int npoints );
      ...

     protected:
      bool computeImpl( const std::vector<Mat>& images, Mat& response );
      ...

    };
@endcode
Example of adding TrackerFeature to TrackerContribFeatureSet : :
@code
    //featureSet is the TrackerContribFeatureSet
    Ptr<TrackerFeature> trackerFeature = new TrackerContribFeatureHAAR( HAARparameters );
    featureSet->addTrackerFeature( trackerFeature );
@endcode
@sa
   TrackerContribFeatureHAAR, TrackerContribFeatureSet

### TrackerModel

TrackerModel is abstract, so in your implementation you must develop your TrackerModel that inherit
from TrackerModel. Fill the method for the estimation of the state "modelEstimationImpl", that
estimates the most likely target location, see @cite AAM table I (ME) for further information. Fill
"modelUpdateImpl" in order to update the model, see @cite AAM table I (MU). In this class you can use
the :cConfidenceMap and :cTrajectory to storing the model. The first represents the model on the all
possible candidate states and the second represents the list of all estimated states.

Example of creating specialized TrackerModel TrackerMILModel : :
@code
    class TrackerMILModel : public TrackerModel
    {
     public:
      TrackerMILModel( const Rect& boundingBox );
      ~TrackerMILModel();
      ...

     protected:
      void modelEstimationImpl( const std::vector<Mat>& responses );
      void modelUpdateImpl();
      ...

    };
@endcode
And add it in your Tracker : :
@code
    bool TrackerMIL::initImpl( const Mat& image, const Rect2d& boundingBox )
    {
      ...
      //model is the general TrackerModel field of the general Tracker
      model = new TrackerMILModel( boundingBox );
      ...
    }
@endcode
In the last step you should define the TrackerStateEstimator based on your implementation or you can
use one of ready class as TrackerStateEstimatorMILBoosting. It represent the statistical part of the
model that estimates the most likely target state.

Example of creating specialized TrackerStateEstimator TrackerStateEstimatorMILBoosting : :
@code
    class CV_EXPORTS_W TrackerStateEstimatorMILBoosting : public TrackerStateEstimator
    {
     class TrackerMILTargetState : public TrackerTargetState
     {
     ...
     };

     public:
      TrackerStateEstimatorMILBoosting( int nFeatures = 250 );
      ~TrackerStateEstimatorMILBoosting();
      ...

     protected:
      Ptr<TrackerTargetState> estimateImpl( const std::vector<ConfidenceMap>& confidenceMaps );
      void updateImpl( std::vector<ConfidenceMap>& confidenceMaps );
      ...

    };
@endcode
And add it in your TrackerModel : :
@code
    //model is the TrackerModel of your Tracker
    Ptr<TrackerStateEstimatorMILBoosting> stateEstimator = new TrackerStateEstimatorMILBoosting( params.featureSetNumFeatures );
    model->setTrackerStateEstimator( stateEstimator );
@endcode
@sa
   TrackerModel, TrackerStateEstimatorMILBoosting, TrackerTargetState

During this step, you should define your TrackerTargetState based on your implementation.
TrackerTargetState base class has only the bounding box (upper-left position, width and height), you
can enrich it adding scale factor, target rotation, etc.

Example of creating specialized TrackerTargetState TrackerMILTargetState : :
@code
    class TrackerMILTargetState : public TrackerTargetState
    {
     public:
      TrackerMILTargetState( const Point2f& position, int targetWidth, int targetHeight, bool foreground, const Mat& features );
      ~TrackerMILTargetState();
      ...

     private:
      bool isTarget;
      Mat targetFeatures;
      ...

    };
@endcode

*/


/************************************ TrackerContribFeature Base Classes ************************************/

/** @brief Abstract base class for TrackerContribFeature that represents the feature.
 */
class CV_EXPORTS TrackerContribFeature : public TrackerFeature
{
 public:
  virtual ~TrackerContribFeature();

  /** @brief Create TrackerContribFeature by tracker feature type
    @param trackerFeatureType The TrackerContribFeature name

    The modes available now:

    -   "HAAR" -- Haar Feature-based

    The modes that will be available soon:

    -   "HOG" -- Histogram of Oriented Gradients features
    -   "LBP" -- Local Binary Pattern features
    -   "FEATURE2D" -- All types of Feature2D
  */
  static Ptr<TrackerContribFeature> create( const String& trackerFeatureType );

  /** @brief Identify most effective features
    @param response Collection of response for the specific TrackerContribFeature
    @param npoints Max number of features

    @note This method modifies the response parameter
     */
  virtual void selection( Mat& response, int npoints ) = 0;

  /** @brief Get the name of the specific TrackerContribFeature
     */
  String getClassName() const;

 protected:
  String className;
};

/** @brief Class that manages the extraction and selection of features

@cite AAM Feature Extraction and Feature Set Refinement (Feature Processing and Feature Selection).
See table I and section III C @cite AMVOT Appearance modelling -\> Visual representation (Table II,
section 3.1 - 3.2)

TrackerContribFeatureSet is an aggregation of TrackerContribFeature

@sa
   TrackerContribFeature

 */
class CV_EXPORTS TrackerContribFeatureSet
{
 public:

  TrackerContribFeatureSet();

  ~TrackerContribFeatureSet();

  /** @brief Extract features from the images collection
    @param images The input images
     */
  void extraction( const std::vector<Mat>& images );

  /** @brief Identify most effective features for all feature types (optional)
     */
  void selection();

  /** @brief Remove outliers for all feature types (optional)
     */
  void removeOutliers();

  /** @brief Add TrackerContribFeature in the collection. Return true if TrackerContribFeature is added, false otherwise
    @param trackerFeatureType The TrackerContribFeature name

    The modes available now:

    -   "HAAR" -- Haar Feature-based

    The modes that will be available soon:

    -   "HOG" -- Histogram of Oriented Gradients features
    -   "LBP" -- Local Binary Pattern features
    -   "FEATURE2D" -- All types of Feature2D

    Example TrackerContribFeatureSet::addTrackerFeature : :
    @code
        //sample usage:

        Ptr<TrackerContribFeature> trackerFeature = ...;
        featureSet->addTrackerFeature( trackerFeature );

        //or add CSC sampler with default parameters
        //featureSet->addTrackerFeature( "HAAR" );
    @endcode
    @note If you use the second method, you must initialize the TrackerContribFeature
     */
  bool addTrackerFeature( String trackerFeatureType );

  /** @overload
    @param feature The TrackerContribFeature class
    */
  bool addTrackerFeature( Ptr<TrackerContribFeature>& feature );

  /** @brief Get the TrackerContribFeature collection (TrackerContribFeature name, TrackerContribFeature pointer)
     */
  const std::vector<std::pair<String, Ptr<TrackerContribFeature> > >& getTrackerFeature() const;

  /** @brief Get the responses

    @note Be sure to call extraction before getResponses Example TrackerContribFeatureSet::getResponses : :
     */
  const std::vector<Mat>& getResponses() const;

 private:

  void clearResponses();
  bool blockAddTrackerFeature;

  std::vector<std::pair<String, Ptr<TrackerContribFeature> > > features;  //list of features
  std::vector<Mat> responses;        //list of response after compute

};


/************************************ TrackerContribSampler Base Classes ************************************/

/** @brief Abstract base class for TrackerContribSamplerAlgorithm that represents the algorithm for the specific
sampler.
 */
class CV_EXPORTS TrackerContribSamplerAlgorithm : public TrackerSamplerAlgorithm
{
 public:
  /**
   * \brief Destructor
   */
  virtual ~TrackerContribSamplerAlgorithm();

  /** @brief Create TrackerContribSamplerAlgorithm by tracker sampler type.
    @param trackerSamplerType The trackerSamplerType name

    The modes available now:

    -   "CSC" -- Current State Center
    -   "CS" -- Current State
     */
  static Ptr<TrackerContribSamplerAlgorithm> create( const String& trackerSamplerType );

  /** @brief Computes the regions starting from a position in an image.

    Return true if samples are computed, false otherwise

    @param image The current frame
    @param boundingBox The bounding box from which regions can be calculated

    @param sample The computed samples @cite AAM Fig. 1 variable Sk
     */
  virtual bool sampling(const Mat& image, const Rect& boundingBox, std::vector<Mat>& sample) CV_OVERRIDE;

  /** @brief Get the name of the specific TrackerContribSamplerAlgorithm
    */
  String getClassName() const;

 protected:
  String className;

  virtual bool samplingImpl( const Mat& image, Rect boundingBox, std::vector<Mat>& sample ) = 0;
};

/**
 * \brief Class that manages the sampler in order to select regions for the update the model of the tracker
 * [AAM] Sampling e Labeling. See table I and section III B
 */

/** @brief Class that manages the sampler in order to select regions for the update the model of the tracker

@cite AAM Sampling e Labeling. See table I and section III B

TrackerContribSampler is an aggregation of TrackerContribSamplerAlgorithm
@sa
   TrackerContribSamplerAlgorithm
 */
class CV_EXPORTS TrackerContribSampler
{
 public:

  /**
   * \brief Constructor
   */
  TrackerContribSampler();

  /**
   * \brief Destructor
   */
  ~TrackerContribSampler();

  /** @brief Computes the regions starting from a position in an image
    @param image The current frame
    @param boundingBox The bounding box from which regions can be calculated
     */
  void sampling( const Mat& image, Rect boundingBox );

  /** @brief Return the collection of the TrackerContribSamplerAlgorithm
    */
  const std::vector<std::pair<String, Ptr<TrackerContribSamplerAlgorithm> > >& getSamplers() const;

  /** @brief Return the samples from all TrackerContribSamplerAlgorithm, @cite AAM Fig. 1 variable Sk
    */
  const std::vector<Mat>& getSamples() const;

  /** @brief Add TrackerContribSamplerAlgorithm in the collection. Return true if sampler is added, false otherwise
    @param trackerSamplerAlgorithmType The TrackerContribSamplerAlgorithm name

    The modes available now:
    -   "CSC" -- Current State Center
    -   "CS" -- Current State
    -   "PF" -- Particle Filtering

    Example TrackerContribSamplerAlgorithm::addTrackerContribSamplerAlgorithm : :
    @code
         TrackerContribSamplerCSC::Params CSCparameters;
         Ptr<TrackerContribSamplerAlgorithm> CSCSampler = new TrackerContribSamplerCSC( CSCparameters );

         if( !sampler->addTrackerSamplerAlgorithm( CSCSampler ) )
           return false;

         //or add CSC sampler with default parameters
         //sampler->addTrackerSamplerAlgorithm( "CSC" );
    @endcode
    @note If you use the second method, you must initialize the TrackerContribSamplerAlgorithm
     */
  bool addTrackerSamplerAlgorithm( String trackerSamplerAlgorithmType );

  /** @overload
    @param sampler The TrackerContribSamplerAlgorithm
    */
  bool addTrackerSamplerAlgorithm( Ptr<TrackerContribSamplerAlgorithm>& sampler );

 private:
  std::vector<std::pair<String, Ptr<TrackerContribSamplerAlgorithm> > > samplers;
  std::vector<Mat> samples;
  bool blockAddTrackerSampler;

  void clearSamples();
};


/** @brief TrackerStateEstimatorAdaBoosting based on ADA-Boosting
 */
class CV_EXPORTS TrackerStateEstimatorAdaBoosting : public TrackerStateEstimator
{
 public:
  /** @brief Implementation of the target state for TrackerAdaBoostingTargetState
    */
  class CV_EXPORTS TrackerAdaBoostingTargetState : public TrackerTargetState
  {

   public:
    /**
     * \brief Constructor
     * \param position Top left corner of the bounding box
     * \param width Width of the bounding box
     * \param height Height of the bounding box
     * \param foreground label for target or background
     * \param responses list of features
     */
    TrackerAdaBoostingTargetState( const Point2f& position, int width, int height, bool foreground, const Mat& responses );

    /**
     * \brief Destructor
     */
    ~TrackerAdaBoostingTargetState()
    {
    }
    ;

    /** @brief Set the features extracted from TrackerContribFeatureSet
    @param responses The features extracted
     */
    void setTargetResponses( const Mat& responses );
    /** @brief Set label: true for target foreground, false for background
    @param foreground Label for background/foreground
     */
    void setTargetFg( bool foreground );
    /** @brief Get the features extracted
     */
    Mat getTargetResponses() const;
    /** @brief Get the label. Return true for target foreground, false for background
    */
    bool isTargetFg() const;

   private:
    bool isTarget;
    Mat targetResponses;

  };

  /** @brief Constructor
    @param numClassifer Number of base classifiers
    @param initIterations Number of iterations in the initialization
    @param nFeatures Number of features/weak classifiers
    @param patchSize tracking rect
    @param ROI initial ROI
     */
  TrackerStateEstimatorAdaBoosting( int numClassifer, int initIterations, int nFeatures, Size patchSize, const Rect& ROI );

  /**
   * \brief Destructor
   */
  ~TrackerStateEstimatorAdaBoosting();

  /** @brief Get the sampling ROI
     */
  Rect getSampleROI() const;

  /** @brief Set the sampling ROI
    @param ROI the sampling ROI
     */
  void setSampleROI( const Rect& ROI );

  /** @brief Set the current confidenceMap
    @param confidenceMap The current :cConfidenceMap
     */
  void setCurrentConfidenceMap( ConfidenceMap& confidenceMap );

  /** @brief Get the list of the selected weak classifiers for the classification step
     */
  std::vector<int> computeSelectedWeakClassifier();

  /** @brief Get the list of the weak classifiers that should be replaced
     */
  std::vector<int> computeReplacedClassifier();

  /** @brief Get the list of the weak classifiers that replace those to be replaced
     */
  std::vector<int> computeSwappedClassifier();

 protected:
  Ptr<TrackerTargetState> estimateImpl( const std::vector<ConfidenceMap>& confidenceMaps ) CV_OVERRIDE;
  void updateImpl( std::vector<ConfidenceMap>& confidenceMaps ) CV_OVERRIDE;

  Ptr<StrongClassifierDirectSelection> boostClassifier;

 private:
  int numBaseClassifier;
  int iterationInit;
  int numFeatures;
  bool trained;
  Size initPatchSize;
  Rect sampleROI;
  std::vector<int> replacedClassifier;
  std::vector<int> swappedClassifier;

  ConfidenceMap currentConfidenceMap;
};


/**
 * \brief TrackerStateEstimator based on SVM
 */
class CV_EXPORTS TrackerStateEstimatorSVM : public TrackerStateEstimator
{
 public:
  TrackerStateEstimatorSVM();
  ~TrackerStateEstimatorSVM();

 protected:
  Ptr<TrackerTargetState> estimateImpl( const std::vector<ConfidenceMap>& confidenceMaps ) CV_OVERRIDE;
  void updateImpl( std::vector<ConfidenceMap>& confidenceMaps ) CV_OVERRIDE;
};



/************************************ Specific TrackerSamplerAlgorithm Classes ************************************/

/** @brief TrackerSampler based on CSC (current state centered), used by MIL algorithm TrackerMIL
 */
class CV_EXPORTS TrackerContribSamplerCSC : public TrackerContribSamplerAlgorithm
{
 public:
  enum
  {
    MODE_INIT_POS = 1,  //!< mode for init positive samples
    MODE_INIT_NEG = 2,  //!< mode for init negative samples
    MODE_TRACK_POS = 3,  //!< mode for update positive samples
    MODE_TRACK_NEG = 4,  //!< mode for update negative samples
    MODE_DETECT = 5   //!< mode for detect samples
  };

  struct CV_EXPORTS Params
  {
    Params();
    float initInRad;        //!< radius for gathering positive instances during init
    float trackInPosRad;    //!< radius for gathering positive instances during tracking
    float searchWinSize;  //!< size of search window
    int initMaxNegNum;      //!< # negative samples to use during init
    int trackMaxPosNum;     //!< # positive samples to use during training
    int trackMaxNegNum;     //!< # negative samples to use during training
  };

  /** @brief Constructor
    @param parameters TrackerContribSamplerCSC parameters TrackerContribSamplerCSC::Params
     */
  TrackerContribSamplerCSC( const TrackerContribSamplerCSC::Params &parameters = TrackerContribSamplerCSC::Params() );

  /** @brief Set the sampling mode of TrackerContribSamplerCSC
    @param samplingMode The sampling mode

    The modes are:

    -   "MODE_INIT_POS = 1" -- for the positive sampling in initialization step
    -   "MODE_INIT_NEG = 2" -- for the negative sampling in initialization step
    -   "MODE_TRACK_POS = 3" -- for the positive sampling in update step
    -   "MODE_TRACK_NEG = 4" -- for the negative sampling in update step
    -   "MODE_DETECT = 5" -- for the sampling in detection step
     */
  void setMode( int samplingMode );

  ~TrackerContribSamplerCSC();

 protected:

  bool samplingImpl(const Mat& image, Rect boundingBox, std::vector<Mat>& sample) CV_OVERRIDE;

 private:

  Params params;
  int mode;
  RNG rng;

  std::vector<Mat> sampleImage( const Mat& img, int x, int y, int w, int h, float inrad, float outrad = 0, int maxnum = 1000000 );
};


/** @brief TrackerContribSampler based on CS (current state), used by algorithm TrackerBoosting
 */
class CV_EXPORTS TrackerSamplerCS : public TrackerContribSamplerAlgorithm
{
 public:
  enum
  {
    MODE_POSITIVE = 1,  //!< mode for positive samples
    MODE_NEGATIVE = 2,  //!< mode for negative samples
    MODE_CLASSIFY = 3  //!< mode for classify samples
  };

  struct CV_EXPORTS Params
  {
    Params();
    float overlap;  //!<overlapping for the search windows
    float searchFactor;  //!<search region parameter
  };
  /** @brief Constructor
    @param parameters TrackerSamplerCS parameters TrackerSamplerCS::Params
     */
  TrackerSamplerCS( const TrackerSamplerCS::Params &parameters = TrackerSamplerCS::Params() );

  /** @brief Set the sampling mode of TrackerSamplerCS
    @param samplingMode The sampling mode

    The modes are:

    -   "MODE_POSITIVE = 1" -- for the positive sampling
    -   "MODE_NEGATIVE = 2" -- for the negative sampling
    -   "MODE_CLASSIFY = 3" -- for the sampling in classification step
     */
  void setMode( int samplingMode );

  ~TrackerSamplerCS();

  bool samplingImpl( const Mat& image, Rect boundingBox, std::vector<Mat>& sample ) CV_OVERRIDE;
  Rect getROI() const;
 private:
  Rect getTrackingROI( float searchFactor );
  Rect RectMultiply( const Rect & rect, float f );
  std::vector<Mat> patchesRegularScan( const Mat& image, Rect trackingROI, Size patchSize );
  void setCheckedROI( Rect imageROI );

  Params params;
  int mode;
  Rect trackedPatch;
  Rect validROI;
  Rect ROI;

};

/** @brief This sampler is based on particle filtering.

In principle, it can be thought of as performing some sort of optimization (and indeed, this
tracker uses opencv's optim module), where tracker seeks to find the rectangle in given frame,
which is the most *"similar"* to the initial rectangle (the one, given through the constructor).

The optimization performed is stochastic and somehow resembles genetic algorithms, where on each new
image received (submitted via TrackerSamplerPF::sampling()) we start with the region bounded by
boundingBox, then generate several "perturbed" boxes, take the ones most similar to the original.
This selection round is repeated several times. At the end, we hope that only the most promising box
remaining, and these are combined to produce the subrectangle of image, which is put as a sole
element in array sample.

It should be noted, that the definition of "similarity" between two rectangles is based on comparing
their histograms. As experiments show, tracker is *not* very succesfull if target is assumed to
strongly change its dimensions.
 */
class CV_EXPORTS TrackerSamplerPF : public TrackerContribSamplerAlgorithm
{
public:
  /** @brief This structure contains all the parameters that can be varied during the course of sampling
    algorithm. Below is the structure exposed, together with its members briefly explained with
    reference to the above discussion on algorithm's working.
 */
  struct CV_EXPORTS Params
  {
    Params();
    int iterationNum; //!< number of selection rounds
    int particlesNum; //!< number of "perturbed" boxes on each round
    double alpha; //!< with each new round we exponentially decrease the amount of "perturbing" we allow (like in simulated annealing)
                  //!< and this very alpha controls how fast annealing happens, ie. how fast perturbing decreases
    Mat_<double> std; //!< initial values for perturbing (1-by-4 array, as each rectangle is given by 4 values -- coordinates of opposite vertices,
                      //!< hence we have 4 values to perturb)
  };
  /** @brief Constructor
    @param chosenRect Initial rectangle, that is supposed to contain target we'd like to track.
    @param parameters
     */
  TrackerSamplerPF(const Mat& chosenRect,const TrackerSamplerPF::Params &parameters = TrackerSamplerPF::Params());
protected:
  bool samplingImpl( const Mat& image, Rect boundingBox, std::vector<Mat>& sample ) CV_OVERRIDE;
private:
  Params params;
  Ptr<MinProblemSolver> _solver;
  Ptr<MinProblemSolver::Function> _function;
};



/************************************ Specific TrackerContribFeature Classes ************************************/

/**
 * \brief TrackerContribFeature based on Feature2D
 */
class CV_EXPORTS TrackerFeatureFeature2d : public TrackerContribFeature
{
 public:

  /**
   * \brief Constructor
   * \param detectorType string of FeatureDetector
   * \param descriptorType string of DescriptorExtractor
   */
  TrackerFeatureFeature2d( String detectorType, String descriptorType );

  ~TrackerFeatureFeature2d() CV_OVERRIDE;

  void selection( Mat& response, int npoints ) CV_OVERRIDE;

 protected:

  bool computeImpl( const std::vector<Mat>& images, Mat& response ) CV_OVERRIDE;

 private:

  std::vector<KeyPoint> keypoints;
};

/**
 * \brief TrackerContribFeature based on HOG
 */
class CV_EXPORTS TrackerFeatureHOG : public TrackerContribFeature
{
 public:

  TrackerFeatureHOG();

  ~TrackerFeatureHOG() CV_OVERRIDE;

  void selection( Mat& response, int npoints ) CV_OVERRIDE;

 protected:

  bool computeImpl( const std::vector<Mat>& images, Mat& response ) CV_OVERRIDE;

};

/** @brief TrackerContribFeature based on HAAR features, used by TrackerMIL and many others algorithms
@note HAAR features implementation is copied from apps/traincascade and modified according to MIL
 */
class CV_EXPORTS TrackerContribFeatureHAAR : public TrackerContribFeature
{
 public:
  struct CV_EXPORTS Params
  {
    Params();
    int numFeatures;  //!< # of rects
    Size rectSize;    //!< rect size
    bool isIntegral;  //!< true if input images are integral, false otherwise
  };

  /** @brief Constructor
    @param parameters TrackerContribFeatureHAAR parameters TrackerContribFeatureHAAR::Params
     */
  TrackerContribFeatureHAAR( const TrackerContribFeatureHAAR::Params &parameters = TrackerContribFeatureHAAR::Params() );

  ~TrackerContribFeatureHAAR() CV_OVERRIDE;

  /** @brief Compute the features only for the selected indices in the images collection
    @param selFeatures indices of selected features
    @param images The images
    @param response Collection of response for the specific TrackerContribFeature
     */
  bool extractSelected( const std::vector<int> selFeatures, const std::vector<Mat>& images, Mat& response );

  /** @brief Identify most effective features
    @param response Collection of response for the specific TrackerContribFeature
    @param npoints Max number of features

    @note This method modifies the response parameter
     */
  void selection( Mat& response, int npoints ) CV_OVERRIDE;

  /** @brief Swap the feature in position source with the feature in position target
  @param source The source position
  @param target The target position
 */
  bool swapFeature( int source, int target );

  /** @brief   Swap the feature in position id with the feature input
  @param id The position
  @param feature The feature
 */
  bool swapFeature( int id, CvHaarEvaluator::FeatureHaar& feature );

  /** @brief Get the feature in position id
    @param id The position
     */
  CvHaarEvaluator::FeatureHaar& getFeatureAt( int id );

 protected:
  bool computeImpl( const std::vector<Mat>& images, Mat& response ) CV_OVERRIDE;

 private:

  Params params;
  Ptr<CvHaarEvaluator> featureEvaluator;
};

/**
 * \brief TrackerContribFeature based on LBP
 */
class CV_EXPORTS TrackerFeatureLBP : public TrackerContribFeature
{
 public:

  TrackerFeatureLBP();

  ~TrackerFeatureLBP();

  void selection( Mat& response, int npoints ) CV_OVERRIDE;

 protected:

  bool computeImpl( const std::vector<Mat>& images, Mat& response ) CV_OVERRIDE;

};

//! @}

}}}  // namespace

#endif // OPENCV_TRACKING_DETAIL_HPP
