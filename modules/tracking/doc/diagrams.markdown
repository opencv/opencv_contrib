Tracking diagrams {#tracking_diagrams}
=================

General diagram
===============

@startuml{tracking_uml_general.png}
  package "Tracker"
  package "TrackerFeature"
  package "TrackerSampler"
  package "TrackerModel"

  Tracker -> TrackerModel: create
  Tracker -> TrackerSampler: create
  Tracker -> TrackerFeature: create
@enduml

Tracker diagram
===============

@startuml{tracking_uml_tracking.png}
  package "Tracker package" #DDDDDD {


  class Algorithm

  class Tracker{
    Ptr<TrackerFeatureSet> featureSet;
    Ptr<TrackerSampler> sampler;
    Ptr<TrackerModel> model;
    ---
    +static Ptr<Tracker> create(const string& trackerType);
    +bool init(const Mat& image, const Rect& boundingBox);
    +bool update(const Mat& image, Rect& boundingBox);
  }
  class Tracker
  note right: Tracker is the general interface for each specialized trackers
  class TrackerMIL{
    +static Ptr<TrackerMIL> createTracker(const TrackerMIL::Params &parameters);
      +virtual ~TrackerMIL();
  }
  class TrackerBoosting{
    +static Ptr<TrackerBoosting> createTracker(const TrackerBoosting::Params &parameters);
      +virtual ~TrackerBoosting();
  }
  Algorithm <|-- Tracker : virtual inheritance
  Tracker <|-- TrackerMIL
  Tracker <|-- TrackerBoosting

  note "Single instance of the Tracker" as N1
  TrackerBoosting .. N1
  TrackerMIL .. N1
  }

@enduml

TrackerFeatureSet diagram
=========================

@startuml{tracking_uml_feature.png}
  package "TrackerFeature package" #DDDDDD {

  class TrackerFeatureSet{
    -vector<pair<string, Ptr<TrackerFeature> > > features
    -vector<Mat> responses
    ...
    TrackerFeatureSet();
    ~TrackerFeatureSet();
    --
    +extraction(const std::vector<Mat>& images);
    +selection();
    +removeOutliers();
    +vector<Mat> response getResponses();
    +vector<pair<string TrackerFeatureType, Ptr<TrackerFeature> > > getTrackerFeatures();
    +bool addTrackerFeature(string trackerFeatureType);
    +bool addTrackerFeature(Ptr<TrackerFeature>& feature);
    -clearResponses();
  }

  class TrackerFeature <<virtual>>{
    static Ptr<TrackerFeature> = create(const string& trackerFeatureType);
    compute(const std::vector<Mat>& images, Mat& response);
    selection(Mat& response, int npoints);
  }
  note bottom: Can be specialized as in table II\nA tracker can use more types of features

  class TrackerFeatureFeature2D{
    -vector<Keypoints> keypoints
    ---
    TrackerFeatureFeature2D(string detectorType, string descriptorType);
    ~TrackerFeatureFeature2D();
    ---
    compute(const std::vector<Mat>& images, Mat& response);
    selection( Mat& response, int npoints);
  }
  class TrackerFeatureHOG{
    TrackerFeatureHOG();
    ~TrackerFeatureHOG();
    ---
    compute(const std::vector<Mat>& images, Mat& response);
    selection(Mat& response, int npoints);
  }

  TrackerFeatureSet *-- TrackerFeature
  TrackerFeature <|-- TrackerFeatureHOG
  TrackerFeature <|-- TrackerFeatureFeature2D


  note "Per readability and simplicity in this diagram\n there are only two TrackerFeature but you\n can considering the implementation of the other TrackerFeature" as N1
  TrackerFeatureHOG .. N1
  TrackerFeatureFeature2D .. N1
  }

@enduml


TrackerModel diagram
====================

@startuml{tracking_uml_model.png}
  package "TrackerModel package" #DDDDDD {

  class Typedef << (T,#FF7700) >>{
    ConfidenceMap
    Trajectory
  }

  class TrackerModel{
    -vector<ConfidenceMap> confidenceMaps;
    -Trajectory trajectory;
    -Ptr<TrackerStateEstimator> stateEstimator;
    ...
    TrackerModel();
    ~TrackerModel();

    +bool setTrackerStateEstimator(Ptr<TrackerStateEstimator> trackerStateEstimator);
    +Ptr<TrackerStateEstimator> getTrackerStateEstimator();

    +void modelEstimation(const vector<Mat>& responses);
    +void modelUpdate();
    +void setLastTargetState(const Ptr<TrackerTargetState> lastTargetState);
    +void runStateEstimator();

    +const vector<ConfidenceMap>& getConfidenceMaps();
    +const ConfidenceMap& getLastConfidenceMap();
  }
  class TrackerTargetState <<virtual>>{
    Point2f targetPosition;
    ---
    Point2f getTargetPosition();
    void setTargetPosition(Point2f position);
  }
  class TrackerTargetState
  note bottom: Each tracker can create own state

  class TrackerStateEstimator <<virtual>>{
    ~TrackerStateEstimator();
    static Ptr<TrackerStateEstimator> create(const String& trackeStateEstimatorType);
    Ptr<TrackerTargetState> estimate(const vector<ConfidenceMap>& confidenceMaps)
    void update(vector<ConfidenceMap>& confidenceMaps)
  }

  class TrackerStateEstimatorSVM{
    TrackerStateEstimatorSVM()
    ~TrackerStateEstimatorSVM()
    Ptr<TrackerTargetState> estimate(const vector<ConfidenceMap>& confidenceMaps)
    void update(vector<ConfidenceMap>& confidenceMaps)
  }
  class TrackerStateEstimatorMILBoosting{
    TrackerStateEstimatorMILBoosting()
    ~TrackerStateEstimatorMILBoosting()
    Ptr<TrackerTargetState> estimate(const vector<ConfidenceMap>& confidenceMaps)
    void update(vector<ConfidenceMap>& confidenceMaps)
  }

  TrackerModel -> TrackerStateEstimator: create
  TrackerModel *-- TrackerTargetState
  TrackerStateEstimator <|-- TrackerStateEstimatorMILBoosting
  TrackerStateEstimator <|-- TrackerStateEstimatorSVM
  }
@enduml

TrackerSampler diagram
======================

@startuml{tracking_uml_sampler.png}
  package "TrackerSampler package" #DDDDDD {

  class TrackerSampler{
    -vector<pair<String, Ptr<TrackerSamplerAlgorithm> > > samplers
    -vector<Mat> samples;
    ...
    TrackerSampler();
    ~TrackerSampler();
    +sampling(const Mat& image, Rect boundingBox);
    +const vector<pair<String, Ptr<TrackerSamplerAlgorithm> > >& getSamplers();
    +const vector<Mat>& getSamples();
    +bool addTrackerSamplerAlgorithm(String trackerSamplerAlgorithmType);
    +bool addTrackerSamplerAlgorithm(Ptr<TrackerSamplerAlgorithm>& sampler);
    ---
    -void clearSamples();
  }

  class TrackerSamplerAlgorithm{
    ~TrackerSamplerAlgorithm();
    +static Ptr<TrackerSamplerAlgorithm> create(const String& trackerSamplerType);
    +bool sampling(const Mat& image, Rect boundingBox, vector<Mat>& sample);
  }
  note bottom: A tracker could sample the target\nor it could sample the target and the background


  class TrackerSamplerCS{
    TrackerSamplerCS();
    ~TrackerSamplerCS();
    +bool sampling(const Mat& image, Rect boundingBox, vector<Mat>& sample);
  }
  class TrackerSamplerCSC{
    TrackerSamplerCSC();
    ~TrackerSamplerCSC();
    +bool sampling(const Mat& image, Rect boundingBox, vector<Mat>& sample);
  }


  }
@enduml

MultiTracker diagram
======================

@startuml{tracking_uml_multiple.png}
  package "MultiTracker"
  package "Tracker"

  MultiTracker -> Tracker: create

  note top of Tracker: Several classes can be generated.
@enduml

@startuml{multi_tracker_uml.png}

  class MultiTracker{
    MultiTracker(const String& trackerType = "" );
    ~MultiTracker();
    +bool add( const Mat& image, const Rect2d& boundingBox );
    +bool add( const String& trackerType, const Mat& image, const Rect2d& boundingBox );
    +bool add(const String& trackerType, const Mat& image, std::vector<Rect2d> boundingBox);
    +bool add(const Mat& image, std::vector<Rect2d> boundingBox);
    +bool update( const Mat& image, std::vector<Rect2d> & boundingBox );
    +std::vector<Rect2d> objects;
    ---
    #std::vector< Ptr<Tracker> > trackerList;
    #String defaultAlgorithm;
  }


@enduml
