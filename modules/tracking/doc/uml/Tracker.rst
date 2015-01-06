Tracker diagram
===============

.. uml::

  ..@startuml
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

  ..@enduml
