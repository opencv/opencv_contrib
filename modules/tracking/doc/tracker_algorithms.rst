Tracker Algorithms
==================

.. highlight:: cpp

The following algorithms are implemented at the moment.

.. [MIL] B Babenko, M-H Yang, and S Belongie, Visual Tracking with Online Multiple Instance Learning, In CVPR, 2009

.. [OLB] H Grabner, M Grabner, and H Bischof, Real-time tracking via on-line boosting, In Proc. BMVC, volume 1, pages 47– 56, 2006

TrackerBoosting
---------------

This is a real-time object tracking based on a novel on-line version of the AdaBoost algorithm.
The classifier uses the surrounding background as negative examples in update step to avoid the drifting problem. The implementation is based on
[OLB]_.

.. ocv:class:: TrackerBoosting

Implementation of TrackerBoosting from :ocv:class:`Tracker`::

    class CV_EXPORTS_W TrackerBoosting : public Tracker
    {
     public:
      void read( const FileNode& fn );
      void write( FileStorage& fs ) const;
      static Ptr<trackerBoosting> createTracker(const trackerBoosting::Params &parameters=trackerBoosting::Params());
      virtual ~trackerBoosting(){};

     protected:
      bool initImpl( const Mat& image, const Rect2d& boundingBox );
      bool updateImpl( const Mat& image, Rect2d& boundingBox );
    };

TrackerBoosting::Params
-----------------------------------------------------------------------

.. ocv:struct:: TrackerBoosting::Params

List of BOOSTING parameters::

   struct CV_EXPORTS Params
   {
    Params();
    int numClassifiers;  //the number of classifiers to use in a OnlineBoosting algorithm
    float samplerOverlap;  //search region parameters to use in a OnlineBoosting algorithm
    float samplerSearchFactor;  // search region parameters to use in a OnlineBoosting algorithm
    int iterationInit;  //the initial iterations
    int featureSetNumFeatures;  // #features

    void read( const FileNode& fn );
    void write( FileStorage& fs ) const;
   };

TrackerBoosting::createTracker
-----------------------------------------------------------------------

Constructor

.. ocv:function:: Ptr<trackerBoosting> TrackerBoosting::createTracker(const trackerBoosting::Params &parameters=trackerBoosting::Params())

    :param parameters: BOOSTING parameters :ocv:struct:`TrackerBoosting::Params`

TrackerMIL
----------

The MIL algorithm trains a classifier in an online manner to separate the object from the background. Multiple Instance Learning avoids the drift problem for a robust tracking. The implementation is based on [MIL]_.

Original code can be found here http://vision.ucsd.edu/~bbabenko/project_miltrack.shtml

.. ocv:class:: TrackerMIL

Implementation of TrackerMIL from :ocv:class:`Tracker`::

    class CV_EXPORTS_W TrackerMIL : public Tracker
    {
     public:
      void read( const FileNode& fn );
      void write( FileStorage& fs ) const;
      static Ptr<trackerMIL> createTracker(const trackerMIL::Params &parameters=trackerMIL::Params());
      virtual ~trackerMIL(){};

     protected:
      bool initImpl( const Mat& image, const Rect2d& boundingBox );
      bool updateImpl( const Mat& image, Rect2d& boundingBox );
    };

TrackerMIL::Params
------------------

.. ocv:struct:: TrackerMIL::Params

List of MIL parameters::

   struct CV_EXPORTS Params
   {
    Params();
    //parameters for sampler
    float samplerInitInRadius;   // radius for gathering positive instances during init
    int samplerInitMaxNegNum;    // # negative samples to use during init
    float samplerSearchWinSize;  // size of search window
    float samplerTrackInRadius;  // radius for gathering positive instances during tracking
    int samplerTrackMaxPosNum;   // # positive samples to use during tracking
    int samplerTrackMaxNegNum;   // # negative samples to use during tracking

    int featureSetNumFeatures;   // # features

    void read( const FileNode& fn );
    void write( FileStorage& fs ) const;
   };

TrackerMIL::createTracker
-------------------------------

Constructor

.. ocv:function:: Ptr<trackerMIL> TrackerMIL::createTracker(const trackerMIL::Params &parameters=trackerMIL::Params())

    :param parameters: MIL parameters :ocv:struct:`TrackerMIL::Params`
