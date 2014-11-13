Latent SVM
===============================================================

Discriminatively Trained Part Based Models for Object Detection
---------------------------------------------------------------

The object detector described below has been initially proposed by
P.F. Felzenszwalb in [Felzenszwalb2010a]_.  It is based on a
Dalal-Triggs detector that uses a single filter on histogram of
oriented gradients (HOG) features to represent an object category.
This detector uses a sliding window approach, where a filter is
applied at all positions and scales of an image. The first
innovation is enriching the Dalal-Triggs model using a
star-structured part-based model defined by a "root" filter
(analogous to the Dalal-Triggs filter) plus a set of parts filters
and associated deformation models. The score of one of star models
at a particular position and scale within an image is the score of
the root filter at the given location plus the sum over parts of the
maximum, over placements of that part, of the part filter score on
its location minus a deformation cost easuring the deviation of the
part from its ideal location relative to the root. Both root and
part filter scores are defined by the dot product between a filter
(a set of weights) and a subwindow of a feature pyramid computed
from the input image. Another improvement is a representation of the
class of models by a mixture of star models. The score of a mixture
model at a particular position and scale is the maximum over
components, of the score of that component model at the given
location. 

The detector was dramatically speeded-up with cascade algorithm
proposed by P.F. Felzenszwalb in [Felzenszwalb2010b]_. The algorithm
prunes partial hypotheses using thresholds on their scores.The basic
idea of the algorithm is to use a hierarchy of models defined by an
ordering of the original model's parts. For a model with (n+1)
parts, including the root, a sequence of (n+1) models is obtained.
The i-th model in this sequence is defined by the first i parts from
the original model. Using this hierarchy, low scoring hypotheses can be
pruned after looking at the best configuration of a subset of the parts.
Hypotheses that score high under a weak model are evaluated further using
a richer model.

In OpenCV there is an C++ implementation of Latent SVM.

.. highlight:: cpp

LSVMDetector
-----------------
.. ocv:class:: LSVMDetector

This is a C++ abstract class, it provides external user API to work with Latent SVM.

LSVMDetector::ObjectDetection
----------------------------------
.. ocv:struct:: LSVMDetector::ObjectDetection

  Structure contains the detection information.

  .. ocv:member:: Rect rect

     bounding box for a detected object

  .. ocv:member:: float score

     confidence level

  .. ocv:member:: int classID

     class (model or detector) ID that detect an object

LSVMDetector::~LSVMDetector
-------------------------------------
Destructor.

.. ocv:function:: LSVMDetector::~LSVMDetector()

LSVMDetector::create
-----------------------
Load the trained models from given ``.xml`` files and return ``cv::Ptr<LSVMDetector>``.

.. ocv:function:: static cv::Ptr<LSVMDetector> LSVMDetector::create( const vector<string>& filenames, const vector<string>& classNames=vector<string>() )

    :param filenames: A set of filenames storing the trained detectors (models). Each file contains one model. See examples of such files here /opencv_extra/testdata/cv/LSVMDetector/models_VOC2007/.

    :param classNames: A set of trained models names. If it's empty then the name of each model will be constructed from the name of file containing the model. E.g. the model stored in "/home/user/cat.xml" will get the name "cat".

LSVMDetector::detect
-------------------------
Find rectangular regions in the given image that are likely to contain objects of loaded classes (models)
and corresponding confidence levels.

.. ocv:function:: void LSVMDetector::detect( const Mat& image, vector<ObjectDetection>& objectDetections, float overlapThreshold=0.5f, int numThreads=-1 )

    :param image: An image.
    :param objectDetections: The detections: rectangulars, scores and class IDs.
    :param overlapThreshold: Threshold for the non-maximum suppression algorithm.
    :param numThreads: Number of threads used in parallel version of the algorithm.

LSVMDetector::getClassNames
--------------------------------
Return the class (model) names that were passed in constructor or method ``load`` or extracted from models filenames in those methods.

.. ocv:function:: const vector<string>& LSVMDetector::getClassNames() const

LSVMDetector::getClassCount
--------------------------------
Return a count of loaded models (classes).

.. ocv:function:: size_t LSVMDetector::getClassCount() const


.. [Felzenszwalb2010a] Felzenszwalb, P. F. and Girshick, R. B. and McAllester, D. and Ramanan, D. *Object Detection with Discriminatively Trained Part Based Models*. PAMI, vol. 32, no. 9, pp. 1627-1645, September 2010
.. [Felzenszwalb2010b] Felzenszwalb, P. F. and Girshick, R. B. and McAllester, D. *Cascade Object Detection with Deformable Part Models*. CVPR 2010, pp. 2241-2248

