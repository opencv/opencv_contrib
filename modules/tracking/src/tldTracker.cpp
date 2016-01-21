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

#include "tldTracker.hpp"


namespace cv
{

	TrackerTLD::Params::Params(){}

	void TrackerTLD::Params::read(const cv::FileNode& /*fn*/){}

	void TrackerTLD::Params::write(cv::FileStorage& /*fs*/) const {}


Ptr<TrackerTLD> TrackerTLD::createTracker(const TrackerTLD::Params &parameters)
{
    return Ptr<tld::TrackerTLDImpl>(new tld::TrackerTLDImpl(parameters));
}

namespace tld
{

TrackerTLDImpl::TrackerTLDImpl(const TrackerTLD::Params &parameters) :
    params( parameters )
{
  isInit = false;
  trackerProxy = Ptr<TrackerProxyImpl<TrackerMedianFlow, TrackerMedianFlow::Params> >
      (new TrackerProxyImpl<TrackerMedianFlow, TrackerMedianFlow::Params>());
}

void TrackerTLDImpl::read(const cv::FileNode& fn)
{
  params.read( fn );
}

void TrackerTLDImpl::write(cv::FileStorage& fs) const
{
  params.write( fs );
}

bool TrackerTLDImpl::initImpl(const Mat& image, const Rect2d& boundingBox)
{
    Mat image_gray;
    trackerProxy->init(image, boundingBox);
    cvtColor( image, image_gray, COLOR_BGR2GRAY );
    data = Ptr<Data>(new Data(boundingBox));
    double scale = data->getScale();
    Rect2d myBoundingBox = boundingBox;
    if( scale > 1.0 )
    {
        Mat image_proxy;
        resize(image_gray, image_proxy, Size(cvRound(image.cols * scale), cvRound(image.rows * scale)), 0, 0, DOWNSCALE_MODE);
        image_proxy.copyTo(image_gray);
        myBoundingBox.x *= scale;
        myBoundingBox.y *= scale;
        myBoundingBox.width *= scale;
        myBoundingBox.height *= scale;
    }
    model = Ptr<TrackerTLDModel>(new TrackerTLDModel(params, image_gray, myBoundingBox, data->getMinSize()));

    data->confident = false;
    data->failedLastTime = false;

    return true;
}

bool TrackerTLDImpl::updateImpl(const Mat& image, Rect2d& boundingBox)
{
    Mat image_gray, image_blurred, imageForDetector;
    cvtColor( image, image_gray, COLOR_BGR2GRAY );
    double scale = data->getScale();
    if( scale > 1.0 )
        resize(image_gray, imageForDetector, Size(cvRound(image.cols*scale), cvRound(image.rows*scale)), 0, 0, DOWNSCALE_MODE);
    else
        imageForDetector = image_gray;
    GaussianBlur(imageForDetector, image_blurred, GaussBlurKernelSize, 0.0);
    TrackerTLDModel* tldModel = ((TrackerTLDModel*)static_cast<TrackerModel*>(model));
    data->frameNum++;
    Mat_<uchar> standardPatch(STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE);
    std::vector<TLDDetector::LabeledPatch> detectorResults;
    //best overlap around 92%
    std::vector<Rect2d> candidates;
    std::vector<double> candidatesRes;
    bool trackerNeedsReInit = false;
	bool DETECT_FLG = false;
    for( int i = 0; i < 2; i++ )
    {
        Rect2d tmpCandid = boundingBox;

		if (i == 1)
		{
#ifdef HAVE_OPENCL
			if (ocl::haveOpenCL())
				DETECT_FLG = tldModel->detector->ocl_detect(imageForDetector, image_blurred, tmpCandid, detectorResults, tldModel->getMinSize());
			else
#endif
				DETECT_FLG = tldModel->detector->detect(imageForDetector, image_blurred, tmpCandid, detectorResults, tldModel->getMinSize());
		}
        if( ( (i == 0) && !data->failedLastTime && trackerProxy->update(image, tmpCandid) ) || ( DETECT_FLG))
        {
            candidates.push_back(tmpCandid);
            if( i == 0 )
                resample(image_gray, tmpCandid, standardPatch);
            else
                resample(imageForDetector, tmpCandid, standardPatch);
            candidatesRes.push_back(tldModel->detector->Sc(standardPatch));
        }
        else
        {
            if( i == 0 )
                trackerNeedsReInit = true;
        }
    }
    std::vector<double>::iterator it = std::max_element(candidatesRes.begin(), candidatesRes.end());

    if( it == candidatesRes.end() )
    {
        data->confident = false;
        data->failedLastTime = true;
        return false;
    }
    else
    {
        boundingBox = candidates[it - candidatesRes.begin()];
        data->failedLastTime = false;
        if( trackerNeedsReInit || it != candidatesRes.begin() )
            trackerProxy->init(image, boundingBox);
    }

#if 1
    if( it != candidatesRes.end() )
        resample(imageForDetector, candidates[it - candidatesRes.begin()], standardPatch);
#endif

    if( *it > CORE_THRESHOLD )
        data->confident = true;

    if( data->confident )
    {
        Pexpert pExpert(imageForDetector, image_blurred, boundingBox, tldModel->detector, params, data->getMinSize());
		Nexpert nExpert(imageForDetector, boundingBox, tldModel->detector, params);
        std::vector<Mat_<uchar> > examplesForModel, examplesForEnsemble;
        examplesForModel.reserve(100); examplesForEnsemble.reserve(100);
        int negRelabeled = 0;
        for( int i = 0; i < (int)detectorResults.size(); i++ )
        {
            bool expertResult;
            if( detectorResults[i].isObject )
            {
                expertResult = nExpert(detectorResults[i].rect);
                if( expertResult != detectorResults[i].isObject )
                    negRelabeled++;
            }
            else
            {
                expertResult = pExpert(detectorResults[i].rect);
            }

            detectorResults[i].shouldBeIntegrated = detectorResults[i].shouldBeIntegrated || (detectorResults[i].isObject != expertResult);
            detectorResults[i].isObject = expertResult;
        }
        tldModel->integrateRelabeled(imageForDetector, image_blurred, detectorResults);
        pExpert.additionalExamples(examplesForModel, examplesForEnsemble);
#ifdef HAVE_OPENCL
        if (ocl::haveOpenCL())
            tldModel->ocl_integrateAdditional(examplesForModel, examplesForEnsemble, true);
        else
#endif
        tldModel->integrateAdditional(examplesForModel, examplesForEnsemble, true);
        examplesForModel.clear(); examplesForEnsemble.clear();
        nExpert.additionalExamples(examplesForModel, examplesForEnsemble);

#ifdef HAVE_OPENCL
        if (ocl::haveOpenCL())
            tldModel->ocl_integrateAdditional(examplesForModel, examplesForEnsemble, false);
        else
#endif
            tldModel->integrateAdditional(examplesForModel, examplesForEnsemble, false);
    }
    else
    {
#ifdef CLOSED_LOOP
        tldModel->integrateRelabeled(imageForDetector, image_blurred, detectorResults);
#endif
    }

    return true;
}


int TrackerTLDImpl::Pexpert::additionalExamples(std::vector<Mat_<uchar> >& examplesForModel, std::vector<Mat_<uchar> >& examplesForEnsemble)
{
    examplesForModel.clear(); examplesForEnsemble.clear();
    examplesForModel.reserve(100); examplesForEnsemble.reserve(100);

    std::vector<Rect2d> closest, scanGrid;
    Mat scaledImg, blurredImg;

    double scale = scaleAndBlur(img_, cvRound(log(1.0 * resultBox_.width / (initSize_.width)) / log(SCALE_STEP)),
            scaledImg, blurredImg, GaussBlurKernelSize, SCALE_STEP);
    TLDDetector::generateScanGrid(img_.rows, img_.cols, initSize_, scanGrid);
    getClosestN(scanGrid, Rect2d(resultBox_.x / scale, resultBox_.y / scale, resultBox_.width / scale, resultBox_.height / scale), 10, closest);

    for( int i = 0; i < (int)closest.size(); i++ )
    {
        for( int j = 0; j < 10; j++ )
        {
            Point2f center;
            Size2f size;
            Mat_<uchar> standardPatch(STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE), blurredPatch(initSize_);
            center.x = (float)(closest[i].x + closest[i].width * (0.5 + rng.uniform(-0.01, 0.01)));
            center.y = (float)(closest[i].y + closest[i].height * (0.5 + rng.uniform(-0.01, 0.01)));
            size.width = (float)(closest[i].width * rng.uniform((double)0.99, (double)1.01));
            size.height = (float)(closest[i].height * rng.uniform((double)0.99, (double)1.01));
            float angle = (float)rng.uniform(-5.0, 5.0);

            for( int y = 0; y < standardPatch.rows; y++ )
            {
                for( int x = 0; x < standardPatch.cols; x++ )
                {
                    standardPatch(x, y) += (uchar)rng.gaussian(5.0);
                }
            }
#ifdef BLUR_AS_VADIM
            GaussianBlur(standardPatch, blurredPatch, GaussBlurKernelSize, 0.0);
            resize(blurredPatch, blurredPatch, initSize_);
#else
            resample(blurredImg, RotatedRect(center, size, angle), blurredPatch);
#endif
            resample(scaledImg, RotatedRect(center, size, angle), standardPatch);
            examplesForModel.push_back(standardPatch);
            examplesForEnsemble.push_back(blurredPatch);
        }
    }
    return 0;
}

bool TrackerTLDImpl::Nexpert::operator()(Rect2d box)
{
    if( overlap(resultBox_, box) < NEXPERT_THRESHOLD )
        return false;
    else
        return true;
}

Data::Data(Rect2d initBox)
{
    double minDim = std::min(initBox.width, initBox.height);
    scale = 20.0 / minDim;
    minSize.width = (int)(initBox.width * 20.0 / minDim);
    minSize.height = (int)(initBox.height * 20.0 / minDim);
    frameNum = 0;
}

void Data::printme(FILE*  port)
{
    dfprintf((port, "Data:\n"));
    dfprintf((port, "\tframeNum = %d\n", frameNum));
    dfprintf((port, "\tconfident = %s\n", confident?"true":"false"));
    dfprintf((port, "\tfailedLastTime = %s\n", failedLastTime?"true":"false"));
    dfprintf((port, "\tminSize = %dx%d\n", minSize.width, minSize.height));
}

}

}
