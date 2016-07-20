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

#ifndef __OPENCV_DNN_LAYERS_DETECTION_OUTPUT_LAYER_HPP__
#define __OPENCV_DNN_LAYERS_DETECTION_OUTPUT_LAYER_HPP__

#include "../precomp.hpp"
#include "caffe.pb.h"

namespace cv
{
namespace dnn
{
class DetectionOutputLayer : public Layer
{
    unsigned _numClasses;
    bool _shareLocation;
    int _numLocClasses;

    int _backgroundLabelId;

    typedef caffe::PriorBoxParameter_CodeType CodeType;
    CodeType _codeType;

    bool _varianceEncodedInTarget;
    int _keepTopK;
    float _confidenceThreshold;

    int _num;
    int _numPriors;

    float _nmsThreshold;
    int _topK;

    static const size_t _numAxes = 4;
    static const std::string _layerName;

public:
    DetectionOutputLayer(LayerParams &params);
    void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);

    void checkInputs(const std::vector<Blob*> &inputs);
    void getCodeType(LayerParams &params);

    template<typename T>
    T getParameter(const LayerParams &params,
                   const std::string &parameterName,
                   const size_t &idx = 0,
                   const bool required = true,
                   const T& defaultValue = T());

    bool getParameterDict(const LayerParams &params,
                          const std::string &parameterName,
                          DictValue& result);

    typedef std::map<int, std::vector<caffe::NormalizedBBox> > LabelBBox;

    // Clip the caffe::NormalizedBBox such that the range for each corner is [0, 1].
    void ClipBBox(const caffe::NormalizedBBox& bbox, caffe::NormalizedBBox* clip_bbox);

    // Decode a bbox according to a prior bbox.
    void DecodeBBox(const caffe::NormalizedBBox& prior_bbox,
                    const std::vector<float>& prior_variance, const CodeType code_type,
                    const bool variance_encoded_in_target, const caffe::NormalizedBBox& bbox,
                    caffe::NormalizedBBox* decode_bbox);

    // Decode a set of bboxes according to a set of prior bboxes.
    void DecodeBBoxes(const std::vector<caffe::NormalizedBBox>& prior_bboxes,
                      const std::vector<std::vector<float> >& prior_variances,
                      const CodeType code_type, const bool variance_encoded_in_target,
                      const std::vector<caffe::NormalizedBBox>& bboxes,
                      std::vector<caffe::NormalizedBBox>* decode_bboxes);

    // Decode all bboxes in a batch.
    void DecodeBBoxesAll(const std::vector<LabelBBox>& all_loc_pred,
                         const std::vector<caffe::NormalizedBBox>& prior_bboxes,
                         const std::vector<std::vector<float> >& prior_variances,
                         const size_t num, const bool share_location,
                         const int num_loc_classes, const int background_label_id,
                         const CodeType code_type, const bool variance_encoded_in_target,
                         std::vector<LabelBBox>* all_decode_bboxes);

    // Get prior bounding boxes from prior_data.
    //    prior_data: 1 x 2 x num_priors * 4 x 1 blob.
    //    num_priors: number of priors.
    //    prior_bboxes: stores all the prior bboxes in the format of caffe::NormalizedBBox.
    //    prior_variances: stores all the variances needed by prior bboxes.
    void GetPriorBBoxes(const float* priorData, const int& numPriors,
                        std::vector<caffe::NormalizedBBox>* priorBBoxes,
                        std::vector<std::vector<float> >* priorVariances);

    // Scale the caffe::NormalizedBBox w.r.t. height and width.
    void ScaleBBox(const caffe::NormalizedBBox& bbox, const int height, const int width,
                   caffe::NormalizedBBox* scale_bbox);

    // Do non maximum suppression given bboxes and scores.
    // Inspired by Piotr Dollar's NMS implementation in EdgeBox.
    // https://goo.gl/jV3JYS
    //    bboxes: a set of bounding boxes.
    //    scores: a set of corresponding confidences.
    //    score_threshold: a threshold used to filter detection results.
    //    nms_threshold: a threshold used in non maximum suppression.
    //    top_k: if not -1, keep at most top_k picked indices.
    //    indices: the kept indices of bboxes after nms.
    void ApplyNMSFast(const std::vector<caffe::NormalizedBBox>& bboxes,
                      const std::vector<float>& scores, const float score_threshold,
                      const float nms_threshold, const int top_k, std::vector<int>* indices);


    // Do non maximum suppression given bboxes and scores.
    //    bboxes: a set of bounding boxes.
    //    scores: a set of corresponding confidences.
    //    threshold: the threshold used in non maximu suppression.
    //    top_k: if not -1, keep at most top_k picked indices.
    //    reuse_overlaps: if true, use and update overlaps; otherwise, always
    //      compute overlap.
    //    overlaps: a temp place to optionally store the overlaps between pairs of
    //      bboxes if reuse_overlaps is true.
    //    indices: the kept indices of bboxes after nms.
    void ApplyNMS(const std::vector<caffe::NormalizedBBox>& bboxes,
                  const std::vector<float>& scores,
                  const float threshold, const int top_k, const bool reuse_overlaps,
                  std::map<int, std::map<int, float> >* overlaps, std::vector<int>* indices);

    void ApplyNMS(const bool* overlapped, const int num, std::vector<int>* indices);

    // Get confidence predictions from conf_data.
    //    conf_data: num x num_preds_per_class * num_classes blob.
    //    num: the number of images.
    //    num_preds_per_class: number of predictions per class.
    //    num_classes: number of classes.
    //    conf_preds: stores the confidence prediction, where each item contains
    //      confidence prediction for an image.
    void GetConfidenceScores(const float* conf_data, const int num,
                             const int num_preds_per_class, const int num_classes,
                             std::vector<std::map<int, std::vector<float> > >* conf_scores);

    // Get confidence predictions from conf_data.
    //    conf_data: num x num_preds_per_class * num_classes blob.
    //    num: the number of images.
    //    num_preds_per_class: number of predictions per class.
    //    num_classes: number of classes.
    //    class_major: if true, data layout is
    //      num x num_classes x num_preds_per_class; otherwise, data layerout is
    //      num x num_preds_per_class * num_classes.
    //    conf_preds: stores the confidence prediction, where each item contains
    //      confidence prediction for an image.
    void GetConfidenceScores(const float* conf_data, const int num,
                             const int num_preds_per_class, const int num_classes,
                             const bool class_major,
                             std::vector<std::map<int, std::vector<float> > >* conf_scores);

    // Get location predictions from loc_data.
    //    loc_data: num x num_preds_per_class * num_loc_classes * 4 blob.
    //    num: the number of images.
    //    num_preds_per_class: number of predictions per class.
    //    num_loc_classes: number of location classes. It is 1 if share_location is
    //      true; and is equal to number of classes needed to predict otherwise.
    //    share_location: if true, all classes share the same location prediction.
    //    loc_preds: stores the location prediction, where each item contains
    //      location prediction for an image.
    void GetLocPredictions(const float* loc_data, const int num,
                           const int num_preds_per_class, const int num_loc_classes,
                           const bool share_location, std::vector<LabelBBox>* loc_preds);

    // Get max scores with corresponding indices.
    //    scores: a set of scores.
    //    threshold: only consider scores higher than the threshold.
    //    top_k: if -1, keep all; otherwise, keep at most top_k.
    //    score_index_vec: store the sorted (score, index) pair.
    void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold,
                          const int top_k, std::vector<std::pair<float, int> >* score_index_vec);

    // Compute the jaccard (intersection over union IoU) overlap between two bboxes.
    float JaccardOverlap(const caffe::NormalizedBBox& bbox1, const caffe::NormalizedBBox& bbox2,
                         const bool normalized = true);

    // Compute the intersection between two bboxes.
    void IntersectBBox(const caffe::NormalizedBBox& bbox1, const caffe::NormalizedBBox& bbox2,
                       caffe::NormalizedBBox* intersect_bbox);

    // Compute bbox size.
    float BBoxSize(const caffe::NormalizedBBox& bbox, const bool normalized = true);
};
}
}
#endif
