Non-free 2D Features Algorithms
=================================

This section describes experimental algorithms for 2d feature detection.

StarFeatureDetector
-------------------
.. ocv:class:: StarFeatureDetector : public FeatureDetector

The class implements the keypoint detector introduced by [Agrawal08]_, synonym of ``StarDetector``.  ::

    class StarFeatureDetector : public FeatureDetector
    {
    public:
        StarFeatureDetector( int maxSize=16, int responseThreshold=30,
                             int lineThresholdProjected = 10,
                             int lineThresholdBinarized=8, int suppressNonmaxSize=5 );
        virtual void read( const FileNode& fn );
        virtual void write( FileStorage& fs ) const;
    protected:
        ...
    };

.. [Agrawal08] Agrawal, M., Konolige, K., & Blas, M. R. (2008). Censure: Center surround extremas for realtime feature detection and matching. In Computer Vision–ECCV 2008 (pp. 102-115). Springer Berlin Heidelberg.
