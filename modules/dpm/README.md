Cascade object detection with deformable part models
====================================================
The object detector described below has been initially proposed by P.F. Felzenszwalb in [1]. It is based on a Dalal-Triggs detector that uses a single filter on histogram of oriented gradients (HOG) features to represent an object category. This detector uses a sliding window approach, where a filter is applied at all positions and scales of an image. The first innovation is enriching the Dalal-Triggs model using a star-structured part-based model defined by a "root" filter (analogous to the Dalal-Triggs filter) plus a set of parts filters and associated deformation models. The score of one of star models at a particular position and scale within an image is the score of the root filter at the given location plus the sum over parts of the maximum, over placements of that part, of the part filter score on its location minus a deformation cost easuring the deviation of the part from its ideal location relative to the root. Both root and part filter scores are defined by the dot product between a filter (a set of weights) and a subwindow of a feature pyramid computed from the input image. Another improvement is a representation of the class of models by a mixture of star models. The score of a mixture model at a particular position and scale is the maximum over components, of the score of that component model at the given location.

The detector was dramatically speeded-up with cascade algorithm proposed by P.F. Felzenszwalb in [2]. The algorithm prunes partial hypotheses using thresholds on their scores. The basic idea of the algorithm is to use a hierarchy of models defined by an ordering of the original model's parts. For a model with (n+1) parts, including the root, a sequence of (n+1) models is obtained. The i-th model in this sequence is defined by the first i parts from the original model.
Using this hierarchy, low scoring hypotheses can be pruned after looking at the best configuration of a subset of the parts. Hypotheses that score high under a weak model are evaluated further using a richer model.

In OpenCV there is an C++ implementation of DPM cascade detector.

Usage
-----
```
// load model from model_path
cv::Ptr<DPMDetector> detector = DPMDetector::create(vector<string>(1, model_path));
// read image from image_path
Mat image = imread(image_path);

// detection
vector<DPMDetector::ObjectDetection> ds;
detector->detect(image, ds);
```

Examples
----------
```
// detect using web camera
./example_dpm_cascade_detect_camera <model_path>

// detect for an image sequence
./example_dpm_cascade_detect_sequence <model_path> <image_dir>
```

References
----------
[1]: P. Felzenszwalb, R. Girshick, D. McAllester, D. Ramanan Object Detection with Discriminatively Trained Part Based Models IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 32, No. 9, Sep. 2010.

[2]: P. Felzenszwalb, R. Girshick, D. McAllester Cascade Object Detection with Deformable Part Models IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2010.
