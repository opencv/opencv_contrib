// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "loop_closure_detection.hpp"

namespace cv{
namespace large_kinfu{

LoopClosureDetectionImpl::LoopClosureDetectionImpl(const String& _modelBin, const String& _modelTxt, const Size& _inputSize, int _backendId, int _targetId)
{
    inputSize = _inputSize;
    CV_Assert(!_modelBin.empty());
    if(_modelTxt.empty())
    {
        net = makePtr<dnn::Net>(dnn::readNet(_modelBin));
    } else{
        net = makePtr<dnn::Net>(dnn::readNet(_modelBin, _modelTxt));
    }

    //Only HF-Net with OpenVINO backend was supported.
    // Pre-trained model can be found at https://1drv.ms/u/s!ApQBoiZSe8Evgolqw23hI8D7lP9mKw?e=ywHAc5.
    //! TODO: HF-Net with OpenCV DNN backend.
    net->setPreferableBackend(_backendId);
    net->setPreferableTarget(_targetId);
    outNameDNN = net->getUnconnectedOutLayersNames();

    KFDataBase = makePtr<KeyFrameDatabase>(maxDatabaseSize);
}

bool LoopClosureDetectionImpl::loopCheck(int& tarSubmapID)
{
    //Calculate the similarity with all pictures in the database.

    // If the KFDataBase is too small, then skip.
    if(KFDataBase->getSize() < minDatabaseSize )
        return false;

    double maxScore = 0;
    int bestId = -1;

    std::vector<int> candidateKFs;

    // Find candidate key frames which similarity are greater than the similarityLow.
    candidateKFs = KFDataBase->getCandidateKF(currentDNNFeature, currentSubmapID, similarityLow, maxScore, bestId);

    CV_LOG_INFO(NULL, "LCD: Best Frame ID = " << bestId<<", similarity = "<<maxScore);

    if( candidateKFs.empty() || maxScore < similarityHigh)
        return false;

    // Remove consecutive keyframes and keyframes from the currentSubmapID.
    std::vector<int> duplicateKFs;
    std::vector<int>::iterator iter = candidateKFs.begin();
    std::vector<int>::iterator iterTemp;
    while (iter != candidateKFs.end() )
    {
        Ptr<KeyFrame> keyFrameDB = KFDataBase->getKeyFrameByID(*iter);

        if(keyFrameDB && keyFrameDB->nextKeyFrameID != -1)
        {
            iterTemp = find(candidateKFs.begin(), candidateKFs.end(), keyFrameDB->nextKeyFrameID);
            if( iterTemp != candidateKFs.end() || keyFrameDB->submapID == currentSubmapID )
            {
                duplicateKFs.push_back(*iterTemp);
            }
        }
        iter++;
    }

    // Delete duplicated KFs.
    for(int deleteID : duplicateKFs)
    {
        iterTemp = find(candidateKFs.begin(), candidateKFs.end(), deleteID);
        if(iterTemp != candidateKFs.end())
        {
            candidateKFs.erase(iterTemp);
        }
    }

    // If all candidate KF from the same submap, then return true.
    int tempSubmapID = -1;
    iter = candidateKFs.begin();

    // If the candidate frame does not belong to the same submapID,
    // it means that it is impossible to specify the target SubmapID.
    while (iter != candidateKFs.end() ) {
        Ptr<KeyFrame> keyFrameDB = KFDataBase->getKeyFrameByID(*iter);
        if(tempSubmapID == -1)
        {
            tempSubmapID = keyFrameDB->submapID;
        }else
        {
            if(tempSubmapID != keyFrameDB->submapID)
                return false;
        }
        iter++;
    }
    // Check whether currentFrame is closed to previous looped Keyframe.
    if(currentFrameID - preLoopedKFID < 20)
        return false;

    if(!candidateKFs.empty())
        bestLoopFrame = KFDataBase->getKeyFrameByID(candidateKFs[0]);
    else
        return false;

    // find target submap ID
    if(bestLoopFrame->submapID == -1 || bestLoopFrame->submapID == currentSubmapID)
    {
        return false;
    }
    else
    {
        tarSubmapID = bestLoopFrame->submapID;
        preLoopedKFID = currentFrameID;
        currentFrameID = -1;

#ifdef HAVE_OPENCV_FEATURES2D
        // ORB Feature Matcher.
        return ORBMather(bestLoopFrame->ORBFeatures, currentORBFeature);
#else
        return true;
#endif

    }
}

bool LoopClosureDetectionImpl::ORBMather(InputArray feature1, InputArray feature2)
{
#ifdef HAVE_OPENCV_FEATURES2D
    std::vector<DMatch> matches;
    ORBmatcher->match(feature1,feature2, matches);
    auto min_max = minmax_element(matches.begin(), matches.end(),
                                  [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    double minDist = min_max.first->distance;

    std::vector<DMatch> goodMatches;
    for (auto &match: matches)
    {
        if (match.distance <= std::max(2 * minDist, 30.0))
        {
            goodMatches.push_back(match);
        }
    }
    if(goodMatches.size() < ORBminMathing)
    {
        CV_LOG_INFO(NULL, "LCD: There are too few ORB matching pairs.");
        return false;
    }
    else
    {
        return true;
    }

#else
    return true;
#endif
}

bool LoopClosureDetectionImpl::addFrame(InputArray _img, const int frameID, const int submapID, int& tarSubmapID)
{

    CV_Assert(!_img.empty());
    currentFrameID = frameID;
    currentSubmapID = submapID;

    Mat img;
    if (_img.isUMat())
    {
        _img.copyTo(img);
    }
    else
    {
        img = _img.getMat();
    }

    // feature Extract.
    processFrame(img, currentDNNFeature, currentKeypoints, currentORBFeature);


    // Key frame filtering.
    bool ifLoop = loopCheck(tarSubmapID);

    // add Frame to KeyFrameDataset.
    if(!ifLoop)
    {
#ifdef HAVE_OPENCV_FEATURES2D
        KFDataBase->addKeyFrame(currentDNNFeature, frameID, submapID, currentKeypoints, currentORBFeature);
#else
        KFDataBase->addKeyFrame(currentDNNFeature, frameID, submapID);
#endif
    }
    return ifLoop;
}

void LoopClosureDetectionImpl::reset()
{
    KFDataBase->reset();
}

void LoopClosureDetectionImpl::processFrame(InputArray img, Mat& DNNfeature, std::vector<KeyPoint>& currentKeypoints, Mat& ORBFeature)
{
    std::vector<Mat> outMats;

    // DNN processing.
    Mat imgBlur, outDNN, outORB;
    imgBlur = img.getMat();
    Mat blob = dnn::blobFromImage(imgBlur, 1.0, inputSize);

    net->setInput(blob);
    net->forward(outMats, outNameDNN);

    outMats[0] /= norm(outMats[0]);
    DNNfeature = outMats[0].clone();

    // ORB process
#ifdef HAVE_OPENCV_FEATURES2D
    ORBdetector->detect(img, currentKeypoints);
    ORBdescriptor->compute(img,currentKeypoints,outORB);
    ORBFeature = outORB.clone();
#endif

}

Ptr<LoopClosureDetection> LoopClosureDetection::create(const String& modelBin, const String& modelTxt, const Size& input_size, int backendId, int targetId)
{
    CV_Assert(!modelBin.empty());
    return makePtr<LoopClosureDetectionImpl>(modelBin, modelTxt, input_size, backendId, targetId);
}

} // namespace large_kinfu
}// namespace cv
