.. highlight:: cpp

Integral Channel Features Detector
==================================

This section describes classes for object detection using WaldBoost and Integral
Channel Features from [Šochman05]_ and [Dollár09]_.

computeChannels
---------------

Compute channels for integral channel features evaluation

.. ocv:function:: void computeChannels(InputArray image, vector<Mat>& channels)

    :param image: image for which channels should be computed
    :param channels: output array for computed channels

FeatureEvaluator
----------------

Feature evaluation interface

.. ocv:class:: FeatureEvaluator : public Algorithm

FeatureEvaluator::setChannels
-----------------------------

Set channels for feature evaluation

.. ocv:function:: void FeatureEvaluator::setChannels(InputArrayOfArrays channels)

    :param channels: array of channels to be set

FeatureEvaluator::setPosition
-----------------------------

Set window position to sample features with shift. By default position is (0, 0).

.. ocv:function:: void FeatureEvaluator::setPosition(Size position)

    :param position: position to be set

FeatureEvaluator::evaluate
--------------------------

Evaluate feature value with given index for current channels and window position.

.. ocv:function:: int FeatureEvaluator::evaluate(size_t feature_ind) const

    :param feature_ind: index of feature to be evaluated

FeatureEvaluator::evaluateAll
-----------------------------

Evaluate all features for current channels and window position.

.. ocv:function:: void FeatureEvaluator::evaluateAll(OutputArray feature_values)

    :param feature_values: matrix-column of evaluated feature values


generateFeatures
----------------

Generate integral features. Returns vector of features.

.. ocv:function:: vector<vector<int> > generateFeatures(Size window_size, const string& type, int count=INT_MAX, int channel_count=10)

    :param window_size: size of window in which features should be evaluated
    :param type: feature type. Can be "icf" or "acf"
    :param count: number of features to generate.
    :param channel_count: number of feature channels

createFeatureEvaluator
----------------------

Construct feature evaluator.

.. ocv:function:: Ptr<FeatureEvaluator> createFeatureEvaluator(const vector<vector<int>>& features, const string& type)

    :param features: features for evaluation
    :param type: feature type. Can be "icf" or "acf"


WaldBoostParams
---------------

Parameters for WaldBoost. weak_count — number of weak learners, alpha — cascade
thresholding param.

::

    struct CV_EXPORTS WaldBoostParams
    {
        int weak_count;
        float alpha;

        WaldBoostParams(): weak_count(100), alpha(0.02f)
        {}
    };

WaldBoost
---------

.. ocv:class:: WaldBoost : public Algorithm

WaldBoost::train
----------------

Train WaldBoost cascade for given data. Returns feature indices chosen for
cascade. Feature enumeration starts from 0.

.. ocv:function:: vector<int> WaldBoost::train(const Mat& data, const Mat& labels)

    :param data: matrix of feature values, size M x N, one feature per row
    :param labels: matrix of samples class labels, size 1 x N. Labels can be from {-1, +1}

WaldBoost::predict
------------------

Predict objects class given object that can compute object features. Returns
unnormed confidence value — measure of confidence that object is from class +1.

.. ocv:function:: float WaldBoost::predict(const Ptr<FeatureEvaluator>& feature_evaluator) const

    :param feature_evaluator: object that can compute features by demand

WaldBoost::write
----------------

Write WaldBoost to FileStorage

.. ocv:function:: void WaldBoost::write(FileStorage& fs)

    :param fs: FileStorage for output

WaldBoost::read
---------------

Write WaldBoost to FileNode

.. ocv:function:: void WaldBoost::read(const FileNode& node)

    :param node: FileNode for reading

createWaldBoost
---------------

Construct WaldBoost object.

.. ocv:function:: Ptr<WaldBoost> createWaldBoost(const WaldBoostParams& params = WaldBoostParams())

ICFDetectorParams
-----------------

Params for ICFDetector training.

::

    struct CV_EXPORTS ICFDetectorParams
    {
        int feature_count;
        int weak_count;
        int model_n_rows;
        int model_n_cols;
        int bg_per_image;
        std::string features_type;
        float alpha;
        bool is_grayscale;
        bool use_fast_log;

        ICFDetectorParams(): feature_count(UINT_MAX), weak_count(100),
            model_n_rows(56), model_n_cols(56), bg_per_image(5), 
            alpha(0.02), is_grayscale(false), use_fast_log(false)
        {}
    };

ICFDetector
-----------

.. ocv:class:: ICFDetector

ICFDetector::train
------------------

Train detector.

.. ocv:function:: void ICFDetector::train(const std::vector<String>& pos_filenames, const std::vector<String>& bg_filenames, ICFDetectorParams params = ICFDetectorParams())

    :param pos_path: path to folder with images of objects (wildcards like ``/my/path/*.png`` are allowed)
    :param bg_path: path to folder with background images
    :param params: parameters for detector training

ICFDetector::detect
-------------------

Detect objects on image.

.. ocv:function:: void ICFDetector::detect(const Mat& image, vector<Rect>& objects, float scaleFactor, Size minSize, Size maxSize, float threshold, int slidingStep, std::vector<float>& values)

.. ocv:function:: detect(const Mat& img, std::vector<Rect>& objects, float minScaleFactor, float maxScaleFactor, float factorStep, float threshold, int slidingStep, std::vector<float>& values)

    :param image: image for detection
    :param objects: output array of bounding boxes
    :param scaleFactor: scale between layers in detection pyramid
    :param minSize: min size of objects in pixels
    :param maxSize: max size of objects in pixels
    :param minScaleFactor: min factor by which the image will be resized
    :param maxScaleFactor: max factor by which the image will be resized
    :param factorStep: scaling factor is incremented each pyramid layer according to this parameter
    :param slidingStep: sliding window step
    :param values: output vector with values of positive samples 

ICFDetector::write
------------------

Write detector to FileStorage.

.. ocv:function:: void ICFDetector::write(FileStorage& fs) const

    :param fs: FileStorage for output

ICFDetector::read
-----------------

Write ICFDetector to FileNode

.. ocv:function:: void ICFDetector::read(const FileNode& node)

    :param node: FileNode for reading


.. [Šochman05] J. Šochman and J. Matas. WaldBoost – Learning for Time Constrained Sequential Detection", CVPR, 2005. The paper is available `online <https://dspace.cvut.cz/bitstream/handle/10467/9494/2005-Waldboost-learning-for-time-constrained-sequential-detection.pdf?sequence=1>`__.

.. [Dollár09] P. Dollár, Z. Tu, P. Perona and S. Belongie. "Integral Channel Features", BMCV 2009. The paper is available `online <http://vision.ucsd.edu/~pdollar/files/papers/DollarBMVC09ChnFtrs.pdf>`__.
