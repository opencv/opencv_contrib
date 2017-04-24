/*M ///////////////////////////////////////////////////////////////////////////////////////
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

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <float.h>
#include <string>
#include <caffe.pb.h>

namespace cv
{
namespace dnn
{

namespace util
{

template <typename T>
std::string to_string(T value)
{
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

template <typename T>
void make_error(const std::string& message1, const T& message2)
{
    std::string error(message1);
    error += std::string(util::to_string<int>(message2));
    CV_Error(Error::StsBadArg, error.c_str());
}

template <typename T>
bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}

}

class DetectionOutputLayerImpl : public DetectionOutputLayer
{
public:
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

    enum { _numAxes = 4 };
    static const std::string _layerName;

    typedef std::map<int, std::vector<caffe::NormalizedBBox> > LabelBBox;

    bool getParameterDict(const LayerParams &params,
                          const std::string &parameterName,
                          DictValue& result)
    {
        if (!params.has(parameterName))
        {
            return false;
        }

        result = params.get(parameterName);
        return true;
    }

    template<typename T>
    T getParameter(const LayerParams &params,
                   const std::string &parameterName,
                   const size_t &idx=0,
                   const bool required=true,
                   const T& defaultValue=T())
    {
        DictValue dictValue;
        bool success = getParameterDict(params, parameterName, dictValue);
        if(!success)
        {
            if(required)
            {
                std::string message = _layerName;
                message += " layer parameter does not contain ";
                message += parameterName;
                message += " parameter.";
                CV_Error(Error::StsBadArg, message);
            }
            else
            {
                return defaultValue;
            }
        }
        return dictValue.get<T>(idx);
    }

    void getCodeType(const LayerParams &params)
    {
        String codeTypeString = params.get<String>("code_type").toLowerCase();
        if (codeTypeString == "corner")
            _codeType = caffe::PriorBoxParameter_CodeType_CORNER;
        else if (codeTypeString == "center_size")
            _codeType = caffe::PriorBoxParameter_CodeType_CENTER_SIZE;
        else
            _codeType = caffe::PriorBoxParameter_CodeType_CORNER;
    }

    DetectionOutputLayerImpl(const LayerParams &params)
    {
        _numClasses = getParameter<unsigned>(params, "num_classes");
        _shareLocation = getParameter<bool>(params, "share_location");
        _numLocClasses = _shareLocation ? 1 : _numClasses;
        _backgroundLabelId = getParameter<int>(params, "background_label_id");
        _varianceEncodedInTarget = getParameter<bool>(params, "variance_encoded_in_target", 0, false, false);
        _keepTopK = getParameter<int>(params, "keep_top_k");
        _confidenceThreshold = getParameter<float>(params, "confidence_threshold", 0, false, -FLT_MAX);
        _topK = getParameter<int>(params, "top_k", 0, false, -1);

        getCodeType(params);

        // Parameters used in nms.
        _nmsThreshold = getParameter<float>(params, "nms_threshold");
        CV_Assert(_nmsThreshold > 0.);

        setParamsFrom(params);
    }

    void checkInputs(const std::vector<Mat*> &inputs)
    {
        for (size_t i = 1; i < inputs.size(); i++)
        {
            CV_Assert(inputs[i]->size == inputs[0]->size);
        }
    }

    void allocate(const std::vector<Mat*> &inputs,
                                        std::vector<Mat> &outputs)
    {
        CV_Assert(inputs.size() > 0);
        CV_Assert(inputs[0]->size[0] == inputs[1]->size[0]);
        _num = inputs[0]->size[0];

        _numPriors = inputs[2]->size[2] / 4;
        CV_Assert((_numPriors * _numLocClasses * 4) == inputs[0]->size[1]);
        CV_Assert(int(_numPriors * _numClasses) == inputs[1]->size[1]);

        // num() and channels() are 1.
        // Since the number of bboxes to be kept is unknown before nms, we manually
        // set it to (fake) 1.
        // Each row is a 7 dimension std::vector, which stores
        // [image_id, label, confidence, xmin, ymin, xmax, ymax]
        int outputShape[] = {1, 1, 1, 7};
        outputs[0].create(4, outputShape, CV_32F);
    }

    void forward(std::vector<Mat*> &inputs,
                                       std::vector<Mat> &outputs)
    {
        const float* locationData = inputs[0]->ptr<float>();
        const float* confidenceData = inputs[1]->ptr<float>();
        const float* priorData = inputs[2]->ptr<float>();

        // Retrieve all location predictions.
        std::vector<LabelBBox> allLocationPredictions;
        GetLocPredictions(locationData, _num, _numPriors, _numLocClasses,
                          _shareLocation, &allLocationPredictions);

        // Retrieve all confidences.
        std::vector<std::map<int, std::vector<float> > > allConfidenceScores;
        GetConfidenceScores(confidenceData, _num, _numPriors, _numClasses,
                            &allConfidenceScores);

        // Retrieve all prior bboxes. It is same within a batch since we assume all
        // images in a batch are of same dimension.
        std::vector<caffe::NormalizedBBox> priorBBoxes;
        std::vector<std::vector<float> > priorVariances;
        GetPriorBBoxes(priorData, _numPriors, &priorBBoxes, &priorVariances);

        // Decode all loc predictions to bboxes.
        std::vector<LabelBBox> allDecodedBBoxes;
        DecodeBBoxesAll(allLocationPredictions, priorBBoxes, priorVariances, _num,
                        _shareLocation, _numLocClasses, _backgroundLabelId,
                        _codeType, _varianceEncodedInTarget, &allDecodedBBoxes);

        int numKept = 0;
        std::vector<std::map<int, std::vector<int> > > allIndices;
        for (int i = 0; i < _num; ++i)
        {
            const LabelBBox& decodeBBoxes = allDecodedBBoxes[i];
            const std::map<int, std::vector<float> >& confidenceScores =
            allConfidenceScores[i];
            std::map<int, std::vector<int> > indices;
            int numDetections = 0;
            for (int c = 0; c < (int)_numClasses; ++c)
            {
                if (c == _backgroundLabelId)
                {
                    // Ignore background class.
                    continue;
                }
                if (confidenceScores.find(c) == confidenceScores.end())
                {
                    // Something bad happened if there are no predictions for current label.
                    util::make_error<int>("Could not find confidence predictions for label ", c);
                }

                const std::vector<float>& scores = confidenceScores.find(c)->second;
                int label = _shareLocation ? -1 : c;
                if (decodeBBoxes.find(label) == decodeBBoxes.end())
                {
                    // Something bad happened if there are no predictions for current label.
                    util::make_error<int>("Could not find location predictions for label ", label);
                    continue;
                }
                const std::vector<caffe::NormalizedBBox>& bboxes =
                decodeBBoxes.find(label)->second;
                ApplyNMSFast(bboxes, scores, _confidenceThreshold, _nmsThreshold,
                             _topK, &(indices[c]));
                numDetections += indices[c].size();
            }
            if (_keepTopK > -1 && numDetections > _keepTopK)
            {
                std::vector<std::pair<float, std::pair<int, int> > > scoreIndexPairs;
                for (std::map<int, std::vector<int> >::iterator it = indices.begin();
                     it != indices.end(); ++it)
                {
                    int label = it->first;
                    const std::vector<int>& labelIndices = it->second;
                    if (confidenceScores.find(label) == confidenceScores.end())
                    {
                        // Something bad happened for current label.
                        util::make_error<int>("Could not find location predictions for label ", label);
                        continue;
                    }
                    const std::vector<float>& scores = confidenceScores.find(label)->second;
                    for (size_t j = 0; j < labelIndices.size(); ++j)
                    {
                        size_t idx = labelIndices[j];
                        CV_Assert(idx < scores.size());
                        scoreIndexPairs.push_back(
                                                  std::make_pair(scores[idx], std::make_pair(label, idx)));
                    }
                }
                // Keep outputs k results per image.
                std::sort(scoreIndexPairs.begin(), scoreIndexPairs.end(),
                          util::SortScorePairDescend<std::pair<int, int> >);
                scoreIndexPairs.resize(_keepTopK);
                // Store the new indices.
                std::map<int, std::vector<int> > newIndices;
                for (size_t j = 0; j < scoreIndexPairs.size(); ++j)
                {
                    int label = scoreIndexPairs[j].second.first;
                    int idx = scoreIndexPairs[j].second.second;
                    newIndices[label].push_back(idx);
                }
                allIndices.push_back(newIndices);
                numKept += _keepTopK;
            }
            else
            {
                allIndices.push_back(indices);
                numKept += numDetections;
            }
        }

        if (numKept == 0)
        {
            CV_ErrorNoReturn(Error::StsError, "Couldn't find any detections");
            return;
        }
        int outputShape[] = {1, 1, numKept, 7};
        outputs[0].create(4, outputShape, CV_32F);
        float* outputsData = outputs[0].ptr<float>();

        int count = 0;
        for (int i = 0; i < _num; ++i)
        {
            const std::map<int, std::vector<float> >& confidenceScores =
            allConfidenceScores[i];
            const LabelBBox& decodeBBoxes = allDecodedBBoxes[i];
            for (std::map<int, std::vector<int> >::iterator it = allIndices[i].begin();
                 it != allIndices[i].end(); ++it)
            {
                int label = it->first;
                if (confidenceScores.find(label) == confidenceScores.end())
                {
                    // Something bad happened if there are no predictions for current label.
                    util::make_error<int>("Could not find confidence predictions for label ", label);
                    continue;
                }
                const std::vector<float>& scores = confidenceScores.find(label)->second;
                int locLabel = _shareLocation ? -1 : label;
                if (decodeBBoxes.find(locLabel) == decodeBBoxes.end())
                {
                    // Something bad happened if there are no predictions for current label.
                    util::make_error<int>("Could not find location predictions for label ", locLabel);
                    continue;
                }
                const std::vector<caffe::NormalizedBBox>& bboxes =
                decodeBBoxes.find(locLabel)->second;
                std::vector<int>& indices = it->second;

                for (size_t j = 0; j < indices.size(); ++j)
                {
                    int idx = indices[j];
                    outputsData[count * 7] = i;
                    outputsData[count * 7 + 1] = label;
                    outputsData[count * 7 + 2] = scores[idx];
                    caffe::NormalizedBBox clipBBox;
                    ClipBBox(bboxes[idx], &clipBBox);
                    outputsData[count * 7 + 3] = clipBBox.xmin();
                    outputsData[count * 7 + 4] = clipBBox.ymin();
                    outputsData[count * 7 + 5] = clipBBox.xmax();
                    outputsData[count * 7 + 6] = clipBBox.ymax();

                    ++count;
                }
            }
        }
    }

    // Compute bbox size.
    float BBoxSize(const caffe::NormalizedBBox& bbox,
                   const bool normalized=true)
    {
        if (bbox.xmax() < bbox.xmin() || bbox.ymax() < bbox.ymin())
        {
            // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
            return 0;
        }
        else
        {
            if (bbox.has_size())
            {
                return bbox.size();
            }
            else
            {
                float width = bbox.xmax() - bbox.xmin();
                float height = bbox.ymax() - bbox.ymin();
                if (normalized)
                {
                    return width * height;
                }
                else
                {
                    // If bbox is not within range [0, 1].
                    return (width + 1) * (height + 1);
                }
            }
        }
    }

    // Clip the caffe::NormalizedBBox such that the range for each corner is [0, 1].
    void ClipBBox(const caffe::NormalizedBBox& bbox,
                  caffe::NormalizedBBox* clipBBox)
    {
        clipBBox->set_xmin(std::max(std::min(bbox.xmin(), 1.f), 0.f));
        clipBBox->set_ymin(std::max(std::min(bbox.ymin(), 1.f), 0.f));
        clipBBox->set_xmax(std::max(std::min(bbox.xmax(), 1.f), 0.f));
        clipBBox->set_ymax(std::max(std::min(bbox.ymax(), 1.f), 0.f));
        clipBBox->clear_size();
        clipBBox->set_size(BBoxSize(*clipBBox));
        clipBBox->set_difficult(bbox.difficult());
    }

    // Decode a bbox according to a prior bbox.
    void DecodeBBox(const caffe::NormalizedBBox& priorBBox, const std::vector<float>& priorVariance,
                    const CodeType codeType, const bool varianceEncodedInTarget,
                    const caffe::NormalizedBBox& bbox, caffe::NormalizedBBox* decodeBBox)
    {
        if (codeType == caffe::PriorBoxParameter_CodeType_CORNER)
        {
            if (varianceEncodedInTarget)
            {
                // variance is encoded in target, we simply need to add the offset
                // predictions.
                decodeBBox->set_xmin(priorBBox.xmin() + bbox.xmin());
                decodeBBox->set_ymin(priorBBox.ymin() + bbox.ymin());
                decodeBBox->set_xmax(priorBBox.xmax() + bbox.xmax());
                decodeBBox->set_ymax(priorBBox.ymax() + bbox.ymax());
            }
            else
            {
                // variance is encoded in bbox, we need to scale the offset accordingly.
                decodeBBox->set_xmin(
                                     priorBBox.xmin() + priorVariance[0] * bbox.xmin());
                decodeBBox->set_ymin(
                                     priorBBox.ymin() + priorVariance[1] * bbox.ymin());
                decodeBBox->set_xmax(
                                     priorBBox.xmax() + priorVariance[2] * bbox.xmax());
                decodeBBox->set_ymax(
                                     priorBBox.ymax() + priorVariance[3] * bbox.ymax());
            }
        }
        else if (codeType == caffe::PriorBoxParameter_CodeType_CENTER_SIZE)
        {
            float priorWidth = priorBBox.xmax() - priorBBox.xmin();
            CV_Assert(priorWidth > 0);

            float priorHeight = priorBBox.ymax() - priorBBox.ymin();
            CV_Assert(priorHeight > 0);

            float priorCenterX = (priorBBox.xmin() + priorBBox.xmax()) / 2.;
            float priorCenterY = (priorBBox.ymin() + priorBBox.ymax()) / 2.;

            float decodeBBoxCenterX, decodeBBoxCenterY;
            float decodeBBoxWidth, decodeBBoxHeight;
            if (varianceEncodedInTarget)
            {
                // variance is encoded in target, we simply need to retore the offset
                // predictions.
                decodeBBoxCenterX = bbox.xmin() * priorWidth + priorCenterX;
                decodeBBoxCenterY = bbox.ymin() * priorHeight + priorCenterY;
                decodeBBoxWidth = exp(bbox.xmax()) * priorWidth;
                decodeBBoxHeight = exp(bbox.ymax()) * priorHeight;
            }
            else
            {
                // variance is encoded in bbox, we need to scale the offset accordingly.
                decodeBBoxCenterX =
                priorVariance[0] * bbox.xmin() * priorWidth + priorCenterX;
                decodeBBoxCenterY =
                priorVariance[1] * bbox.ymin() * priorHeight + priorCenterY;
                decodeBBoxWidth =
                exp(priorVariance[2] * bbox.xmax()) * priorWidth;
                decodeBBoxHeight =
                exp(priorVariance[3] * bbox.ymax()) * priorHeight;
            }

            decodeBBox->set_xmin(decodeBBoxCenterX - decodeBBoxWidth / 2.);
            decodeBBox->set_ymin(decodeBBoxCenterY - decodeBBoxHeight / 2.);
            decodeBBox->set_xmax(decodeBBoxCenterX + decodeBBoxWidth / 2.);
            decodeBBox->set_ymax(decodeBBoxCenterY + decodeBBoxHeight / 2.);
        }
        else
        {
            CV_Error(Error::StsBadArg, "Unknown LocLossType.");
        }
        float bboxSize = BBoxSize(*decodeBBox);
        decodeBBox->set_size(bboxSize);
    }

    // Decode a set of bboxes according to a set of prior bboxes.
    void DecodeBBoxes(const std::vector<caffe::NormalizedBBox>& priorBBoxes,
                      const std::vector<std::vector<float> >& priorVariances,
                      const CodeType codeType, const bool varianceEncodedInTarget,
                      const std::vector<caffe::NormalizedBBox>& bboxes,
                      std::vector<caffe::NormalizedBBox>* decodeBBoxes)
    {
        CV_Assert(priorBBoxes.size() == priorVariances.size());
        CV_Assert(priorBBoxes.size() == bboxes.size());
        int numBBoxes = priorBBoxes.size();
        if (numBBoxes >= 1)
        {
            CV_Assert(priorVariances[0].size() == 4);
        }
        decodeBBoxes->clear();
        for (int i = 0; i < numBBoxes; ++i)
        {
            caffe::NormalizedBBox decodeBBox;
            DecodeBBox(priorBBoxes[i], priorVariances[i], codeType,
                       varianceEncodedInTarget, bboxes[i], &decodeBBox);
            decodeBBoxes->push_back(decodeBBox);
        }
    }

    // Decode all bboxes in a batch.
    void DecodeBBoxesAll(const std::vector<LabelBBox>& allLocPreds,
                         const std::vector<caffe::NormalizedBBox>& priorBBoxes,
                         const std::vector<std::vector<float> >& priorVariances,
                         const size_t num, const bool shareLocation,
                         const int numLocClasses, const int backgroundLabelId,
                         const CodeType codeType, const bool varianceEncodedInTarget,
                         std::vector<LabelBBox>* allDecodeBBoxes)
    {
        CV_Assert(allLocPreds.size() == num);
        allDecodeBBoxes->clear();
        allDecodeBBoxes->resize(num);
        for (size_t i = 0; i < num; ++i)
        {
            // Decode predictions into bboxes.
            LabelBBox& decodeBBoxes = (*allDecodeBBoxes)[i];
            for (int c = 0; c < numLocClasses; ++c)
            {
                int label = shareLocation ? -1 : c;
                if (label == backgroundLabelId)
                {
                    // Ignore background class.
                    continue;
                }
                if (allLocPreds[i].find(label) == allLocPreds[i].end())
                {
                    // Something bad happened if there are no predictions for current label.
                    util::make_error<int>("Could not find location predictions for label ", label);
                }
                const std::vector<caffe::NormalizedBBox>& labelLocPreds =
                allLocPreds[i].find(label)->second;
                DecodeBBoxes(priorBBoxes, priorVariances,
                             codeType, varianceEncodedInTarget,
                             labelLocPreds, &(decodeBBoxes[label]));
            }
        }
    }

    // Get prior bounding boxes from prior_data.
    //    prior_data: 1 x 2 x num_priors * 4 x 1 blob.
    //    num_priors: number of priors.
    //    prior_bboxes: stores all the prior bboxes in the format of caffe::NormalizedBBox.
    //    prior_variances: stores all the variances needed by prior bboxes.
    void GetPriorBBoxes(const float* priorData, const int& numPriors,
                        std::vector<caffe::NormalizedBBox>* priorBBoxes,
                        std::vector<std::vector<float> >* priorVariances)
    {
        priorBBoxes->clear();
        priorVariances->clear();
        for (int i = 0; i < numPriors; ++i)
        {
            int startIdx = i * 4;
            caffe::NormalizedBBox bbox;
            bbox.set_xmin(priorData[startIdx]);
            bbox.set_ymin(priorData[startIdx + 1]);
            bbox.set_xmax(priorData[startIdx + 2]);
            bbox.set_ymax(priorData[startIdx + 3]);
            float bboxSize = BBoxSize(bbox);
            bbox.set_size(bboxSize);
            priorBBoxes->push_back(bbox);
        }

        for (int i = 0; i < numPriors; ++i)
        {
            int startIdx = (numPriors + i) * 4;
            std::vector<float> var;
            for (int j = 0; j < 4; ++j)
            {
                var.push_back(priorData[startIdx + j]);
            }
            priorVariances->push_back(var);
        }
    }

    // Scale the caffe::NormalizedBBox w.r.t. height and width.
    void ScaleBBox(const caffe::NormalizedBBox& bbox,
                   const int height, const int width,
                   caffe::NormalizedBBox* scaleBBox)
    {
        scaleBBox->set_xmin(bbox.xmin() * width);
        scaleBBox->set_ymin(bbox.ymin() * height);
        scaleBBox->set_xmax(bbox.xmax() * width);
        scaleBBox->set_ymax(bbox.ymax() * height);
        scaleBBox->clear_size();
        bool normalized = !(width > 1 || height > 1);
        scaleBBox->set_size(BBoxSize(*scaleBBox, normalized));
        scaleBBox->set_difficult(bbox.difficult());
    }

    // Get location predictions from loc_data.
    //    loc_data: num x num_preds_per_class * num_loc_classes * 4 blob.
    //    num: the number of images.
    //    num_preds_per_class: number of predictions per class.
    //    num_loc_classes: number of location classes. It is 1 if share_location is
    //      true; and is equal to number of classes needed to predict otherwise.
    //    share_location: if true, all classes share the same location prediction.
    //    loc_preds: stores the location prediction, where each item contains
    //      location prediction for an image.
    void GetLocPredictions(const float* locData, const int num,
                           const int numPredsPerClass, const int numLocClasses,
                           const bool shareLocation, std::vector<LabelBBox>* locPreds)
    {
        locPreds->clear();
        if (shareLocation)
        {
            CV_Assert(numLocClasses == 1);
        }
        locPreds->resize(num);
        for (int i = 0; i < num; ++i)
        {
            LabelBBox& labelBBox = (*locPreds)[i];
            for (int p = 0; p < numPredsPerClass; ++p)
            {
                int startIdx = p * numLocClasses * 4;
                for (int c = 0; c < numLocClasses; ++c)
                {
                    int label = shareLocation ? -1 : c;
                    if (labelBBox.find(label) == labelBBox.end())
                    {
                        labelBBox[label].resize(numPredsPerClass);
                    }
                    labelBBox[label][p].set_xmin(locData[startIdx + c * 4]);
                    labelBBox[label][p].set_ymin(locData[startIdx + c * 4 + 1]);
                    labelBBox[label][p].set_xmax(locData[startIdx + c * 4 + 2]);
                    labelBBox[label][p].set_ymax(locData[startIdx + c * 4 + 3]);
                }
            }
            locData += numPredsPerClass * numLocClasses * 4;
        }
    }

    // Get confidence predictions from conf_data.
    //    conf_data: num x num_preds_per_class * num_classes blob.
    //    num: the number of images.
    //    num_preds_per_class: number of predictions per class.
    //    num_classes: number of classes.
    //    conf_preds: stores the confidence prediction, where each item contains
    //      confidence prediction for an image.
    void GetConfidenceScores(const float* confData, const int num,
                             const int numPredsPerClass, const int numClasses,
                             std::vector<std::map<int, std::vector<float> > >* confPreds)
    {
        confPreds->clear();
        confPreds->resize(num);
        for (int i = 0; i < num; ++i)
        {
            std::map<int, std::vector<float> >& labelScores = (*confPreds)[i];
            for (int p = 0; p < numPredsPerClass; ++p)
            {
                int startIdx = p * numClasses;
                for (int c = 0; c < numClasses; ++c)
                {
                    labelScores[c].push_back(confData[startIdx + c]);
                }
            }
            confData += numPredsPerClass * numClasses;
        }
    }

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
                      const std::vector<float>& scores,
                      const float score_threshold,
                      const float nms_threshold, const int top_k,
                      std::vector<int>* indices)
    {
        // Sanity check.
        CV_Assert(bboxes.size() == scores.size());

        // Get top_k scores (with corresponding indices).
        std::vector<std::pair<float, int> > score_index_vec;
        GetMaxScoreIndex(scores, score_threshold, top_k, &score_index_vec);

        // Do nms.
        indices->clear();
        while (score_index_vec.size() != 0)
        {
            const int idx = score_index_vec.front().second;
            bool keep = true;
            for (size_t k = 0; k < indices->size(); ++k)
            {
                if (keep)
                {
                    const int kept_idx = (*indices)[k];
                    float overlap = JaccardOverlap(bboxes[idx], bboxes[kept_idx]);
                    keep = overlap <= nms_threshold;
                }
                else
                {
                    break;
                }
            }
            if (keep)
            {
                indices->push_back(idx);
            }
            score_index_vec.erase(score_index_vec.begin());
        }
    }

    // Get max scores with corresponding indices.
    //    scores: a set of scores.
    //    threshold: only consider scores higher than the threshold.
    //    top_k: if -1, keep all; otherwise, keep at most top_k.
    //    score_index_vec: store the sorted (score, index) pair.
    void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold,const int top_k,
                          std::vector<std::pair<float, int> >* score_index_vec)
    {
        // Generate index score pairs.
        for (size_t i = 0; i < scores.size(); ++i)
        {
            if (scores[i] > threshold)
            {
                score_index_vec->push_back(std::make_pair(scores[i], i));
            }
        }

        // Sort the score pair according to the scores in descending order
        std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
                         util::SortScorePairDescend<int>);

        // Keep top_k scores if needed.
        if (top_k > -1 && top_k < (int)score_index_vec->size())
        {
            score_index_vec->resize(top_k);
        }
    }

    // Compute the intersection between two bboxes.
    void IntersectBBox(const caffe::NormalizedBBox& bbox1,
                       const caffe::NormalizedBBox& bbox2,
                       caffe::NormalizedBBox* intersect_bbox) {
        if (bbox2.xmin() > bbox1.xmax() || bbox2.xmax() < bbox1.xmin() ||
            bbox2.ymin() > bbox1.ymax() || bbox2.ymax() < bbox1.ymin())
        {
            // Return [0, 0, 0, 0] if there is no intersection.
            intersect_bbox->set_xmin(0);
            intersect_bbox->set_ymin(0);
            intersect_bbox->set_xmax(0);
            intersect_bbox->set_ymax(0);
        }
        else
        {
            intersect_bbox->set_xmin(std::max(bbox1.xmin(), bbox2.xmin()));
            intersect_bbox->set_ymin(std::max(bbox1.ymin(), bbox2.ymin()));
            intersect_bbox->set_xmax(std::min(bbox1.xmax(), bbox2.xmax()));
            intersect_bbox->set_ymax(std::min(bbox1.ymax(), bbox2.ymax()));
        }
    }

    // Compute the jaccard (intersection over union IoU) overlap between two bboxes.
    float JaccardOverlap(const caffe::NormalizedBBox& bbox1,
                         const caffe::NormalizedBBox& bbox2,
                         const bool normalized=true)
    {
        caffe::NormalizedBBox intersect_bbox;
        IntersectBBox(bbox1, bbox2, &intersect_bbox);
        float intersect_width, intersect_height;
        if (normalized)
        {
            intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin();
            intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin();
        }
        else
        {
            intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin() + 1;
            intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin() + 1;
        }
        if (intersect_width > 0 && intersect_height > 0)
        {
            float intersect_size = intersect_width * intersect_height;
            float bbox1_size = BBoxSize(bbox1);
            float bbox2_size = BBoxSize(bbox2);
            return intersect_size / (bbox1_size + bbox2_size - intersect_size);
        }
        else
        {
            return 0.;
        }
    }
};

const std::string DetectionOutputLayerImpl::_layerName = std::string("DetectionOutput");

Ptr<DetectionOutputLayer> DetectionOutputLayer::create(const LayerParams &params)
{
    return Ptr<DetectionOutputLayer>(new DetectionOutputLayerImpl(params));
}

}
}
