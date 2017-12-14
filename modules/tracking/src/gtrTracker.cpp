/*///////////////////////////////////////////////////////////////////////////////////////
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
#include "opencv2/opencv_modules.hpp"
#include "gtrTracker.hpp"


namespace cv
{

TrackerGOTURN::Params::Params(){}

void TrackerGOTURN::Params::read(const cv::FileNode& /*fn*/){}

void TrackerGOTURN::Params::write(cv::FileStorage& /*fs*/) const {}


Ptr<TrackerGOTURN> TrackerGOTURN::create(const TrackerGOTURN::Params &parameters)
{
#ifdef HAVE_OPENCV_DNN
    return Ptr<gtr::TrackerGOTURNImpl>(new gtr::TrackerGOTURNImpl(parameters));
#else
    (void)(parameters);
    CV_ErrorNoReturn(cv::Error::StsNotImplemented , "to use GOTURN, the tracking module needs to be built with opencv_dnn !");
#endif
}
Ptr<TrackerGOTURN> TrackerGOTURN::create()
{
    return TrackerGOTURN::create(TrackerGOTURN::Params());
}


#ifdef HAVE_OPENCV_DNN
namespace gtr
{

class TrackerGOTURNModel : public TrackerModel{
public:
    TrackerGOTURNModel(TrackerGOTURN::Params){}
    Rect2d getBoundingBox(){ return boundingBox_; }
    void setBoudingBox(Rect2d boundingBox){ boundingBox_ = boundingBox; }
    Mat getImage(){ return image_; }
    void setImage(const Mat& image){ image.copyTo(image_); }
protected:
    Rect2d boundingBox_;
    Mat image_;
    void modelEstimationImpl(const std::vector<Mat>&){}
    void modelUpdateImpl(){}
};

TrackerGOTURNImpl::TrackerGOTURNImpl(const TrackerGOTURN::Params &parameters) :
    params(parameters){
    isInit = false;
};

void TrackerGOTURNImpl::read(const cv::FileNode& fn)
{
    params.read(fn);
}

void TrackerGOTURNImpl::write(cv::FileStorage& fs) const
{
    params.write(fs);
}

bool TrackerGOTURNImpl::initImpl(const Mat& image, const Rect2d& boundingBox)
{
    //Make a simple model from frame and bounding box
    model = Ptr<TrackerGOTURNModel>(new TrackerGOTURNModel(params));
    ((TrackerGOTURNModel*)static_cast<TrackerModel*>(model))->setImage(image);
    ((TrackerGOTURNModel*)static_cast<TrackerModel*>(model))->setBoudingBox(boundingBox);

    //Load GOTURN architecture from *.prototxt and pretrained weights from *.caffemodel
    String modelTxt = "goturn.prototxt";
    String modelBin = "goturn.caffemodel";
    net = dnn::readNetFromCaffe(modelTxt, modelBin);
    return true;
}

bool TrackerGOTURNImpl::updateImpl(const Mat& image, Rect2d& boundingBox)
{
    int INPUT_SIZE = 227;
    //Using prevFrame & prevBB from model and curFrame GOTURN calculating curBB
    Mat curFrame = image.clone();
    Mat prevFrame = ((TrackerGOTURNModel*)static_cast<TrackerModel*>(model))->getImage();
    Rect2d prevBB = ((TrackerGOTURNModel*)static_cast<TrackerModel*>(model))->getBoundingBox();
    Rect2d curBB;

    float padTargetPatch = 2.0;
    Rect2f searchPatchRect, targetPatchRect;
    Point2f currCenter, prevCenter;
    Mat prevFramePadded, curFramePadded;
    Mat searchPatch, targetPatch;

    prevCenter.x = (float)(prevBB.x + prevBB.width / 2);
    prevCenter.y = (float)(prevBB.y + prevBB.height / 2);

    targetPatchRect.width = (float)(prevBB.width*padTargetPatch);
    targetPatchRect.height = (float)(prevBB.height*padTargetPatch);
    targetPatchRect.x = (float)(prevCenter.x - prevBB.width*padTargetPatch / 2.0 + targetPatchRect.width);
    targetPatchRect.y = (float)(prevCenter.y - prevBB.height*padTargetPatch / 2.0 + targetPatchRect.height);

    copyMakeBorder(prevFrame, prevFramePadded, (int)targetPatchRect.height, (int)targetPatchRect.height, (int)targetPatchRect.width, (int)targetPatchRect.width, BORDER_REPLICATE);
    targetPatch = prevFramePadded(targetPatchRect).clone();

    copyMakeBorder(curFrame, curFramePadded, (int)targetPatchRect.height, (int)targetPatchRect.height, (int)targetPatchRect.width, (int)targetPatchRect.width, BORDER_REPLICATE);
    searchPatch = curFramePadded(targetPatchRect).clone();

    //Preprocess
    //Resize
    resize(targetPatch, targetPatch, Size(INPUT_SIZE, INPUT_SIZE), 0, 0, INTER_LINEAR_EXACT);
    resize(searchPatch, searchPatch, Size(INPUT_SIZE, INPUT_SIZE), 0, 0, INTER_LINEAR_EXACT);

    //Mean Subtract
    targetPatch = targetPatch - 128;
    searchPatch = searchPatch - 128;

    //Convert to Float type
    Mat targetBlob = dnn::blobFromImage(targetPatch, 1.0f, Size(), Scalar(), false);
    Mat searchBlob = dnn::blobFromImage(searchPatch, 1.0f, Size(), Scalar(), false);

    net.setInput(targetBlob, "data1");
    net.setInput(searchBlob, "data2");

    Mat resMat = net.forward("scale").reshape(1, 1);

    curBB.x = targetPatchRect.x + (resMat.at<float>(0) * targetPatchRect.width / INPUT_SIZE) - targetPatchRect.width;
    curBB.y = targetPatchRect.y + (resMat.at<float>(1) * targetPatchRect.height / INPUT_SIZE) - targetPatchRect.height;
    curBB.width = (resMat.at<float>(2) - resMat.at<float>(0)) * targetPatchRect.width / INPUT_SIZE;
    curBB.height = (resMat.at<float>(3) - resMat.at<float>(1)) * targetPatchRect.height / INPUT_SIZE;

    //Predicted BB
    boundingBox = curBB;

    //Set new model image and BB from current frame
    ((TrackerGOTURNModel*)static_cast<TrackerModel*>(model))->setImage(curFrame);
    ((TrackerGOTURNModel*)static_cast<TrackerModel*>(model))->setBoudingBox(curBB);

    return true;
}

}
#endif // OPENCV_HAVE_DNN

}
