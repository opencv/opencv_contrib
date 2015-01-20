Experimental 2D Features Algorithms
===================================

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


BriefDescriptorExtractor
------------------------
.. ocv:class:: BriefDescriptorExtractor : public DescriptorExtractor

Class for computing BRIEF descriptors described in a paper of Calonder M., Lepetit V.,
Strecha C., Fua P. *BRIEF: Binary Robust Independent Elementary Features* ,
11th European Conference on Computer Vision (ECCV), Heraklion, Crete. LNCS Springer, September 2010. ::

    class BriefDescriptorExtractor : public DescriptorExtractor
    {
    public:
        static const int PATCH_SIZE = 48;
        static const int KERNEL_SIZE = 9;

        // bytes is a length of descriptor in bytes. It can be equal 16, 32 or 64 bytes.
        BriefDescriptorExtractor( int bytes = 32 );

        virtual void read( const FileNode& );
        virtual void write( FileStorage& ) const;
        virtual int descriptorSize() const;
        virtual int descriptorType() const;
        virtual int defaultNorm() const;
    protected:
        ...
    };

.. note::

   * A complete BRIEF extractor sample can be found at opencv_source_code/samples/cpp/brief_match_test.cpp


FREAK
-----
.. ocv:class:: FREAK : public DescriptorExtractor

Class implementing the FREAK (*Fast Retina Keypoint*) keypoint descriptor, described in [AOV12]_. The algorithm propose a novel keypoint descriptor inspired by the human visual system and more precisely the retina, coined Fast Retina Key- point (FREAK). A cascade of binary strings is computed by efficiently comparing image intensities over a retinal sampling pattern. FREAKs are in general faster to compute with lower memory load and also more robust than SIFT, SURF or BRISK. They are competitive alternatives to existing keypoints in particular for embedded applications.

.. [AOV12] A. Alahi, R. Ortiz, and P. Vandergheynst. FREAK: Fast Retina Keypoint. In IEEE Conference on Computer Vision and Pattern Recognition, 2012. CVPR 2012 Open Source Award Winner.

.. note::

   * An example on how to use the FREAK descriptor can be found at opencv_source_code/samples/cpp/freak_demo.cpp

FREAK::FREAK
------------
The FREAK constructor

.. ocv:function:: FREAK::FREAK( bool orientationNormalized=true, bool scaleNormalized=true, float patternScale=22.0f, int nOctaves=4, const vector<int>& selectedPairs=vector<int>() )

    :param orientationNormalized: Enable orientation normalization.
    :param scaleNormalized: Enable scale normalization.
    :param patternScale: Scaling of the description pattern.
    :param nOctaves: Number of octaves covered by the detected keypoints.
    :param selectedPairs: (Optional) user defined selected pairs indexes,

FREAK::selectPairs
------------------
Select the 512 best description pair indexes from an input (grayscale) image set. FREAK is available with a set of pairs learned off-line. Researchers can run a training process to learn their own set of pair. For more details read section 4.2 in: A. Alahi, R. Ortiz, and P. Vandergheynst. FREAK: Fast Retina Keypoint. In IEEE Conference on Computer Vision and Pattern Recognition, 2012.

We notice that for keypoint matching applications, image content has little effect on the selected pairs unless very specific what does matter is the detector type (blobs, corners,...) and the options used (scale/rotation invariance,...). Reduce corrThresh if not enough pairs are selected (43 points --> 903 possible pairs)

.. ocv:function:: vector<int> FREAK::selectPairs(const vector<Mat>& images, vector<vector<KeyPoint> >& keypoints, const double corrThresh = 0.7, bool verbose = true)

    :param images: Grayscale image input set.
    :param keypoints: Set of detected keypoints
    :param corrThresh: Correlation threshold.
    :param verbose: Prints pair selection informations.
