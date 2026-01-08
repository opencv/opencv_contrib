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

#ifndef OPENCV_TLD_TRACKER
#define OPENCV_TLD_TRACKER

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "tldModel.hpp"
#include <algorithm>
#include <limits.h>

namespace cv {
inline namespace tracking {
namespace impl {
namespace tld {

class TrackerProxy
{
public:
	virtual bool init(const Mat& image, const Rect2d& boundingBox) = 0;
	virtual bool update(const Mat& image, Rect2d& boundingBox) = 0;
	virtual ~TrackerProxy(){}
};


class MyMouseCallbackDEBUG
{
public:
	MyMouseCallbackDEBUG(Mat& img, Mat& imgBlurred, TLDDetector* detector) :img_(img), imgBlurred_(imgBlurred), detector_(detector){}
	static void onMouse(int event, int x, int y, int, void* obj){ ((MyMouseCallbackDEBUG*)obj)->onMouse(event, x, y); }
	MyMouseCallbackDEBUG& operator = (const MyMouseCallbackDEBUG& /*other*/){ return *this; }
private:
	void onMouse(int event, int x, int y);
	Mat& img_, imgBlurred_;
	TLDDetector* detector_;
};


class Data
{
public:
	Data(Rect2d initBox);
	Size getMinSize(){ return minSize; }
	double getScale(){ return scale; }
	bool confident;
	bool failedLastTime;
	int frameNum;
	void printme(FILE*  port = stdout);
private:
	double scale;
	Size minSize;
};

template<class T, class Tparams>
class TrackerProxyImpl : public TrackerProxy
{
public:
	TrackerProxyImpl(Tparams params = Tparams()) :params_(params){}
	bool init(const Mat& image, const Rect2d& boundingBox) CV_OVERRIDE
	{
        trackerPtr = T::create();
		return trackerPtr->init(image, boundingBox);
	}
	bool update(const Mat& image, Rect2d& boundingBox) CV_OVERRIDE
	{
		return trackerPtr->update(image, boundingBox);
	}
private:
	Ptr<T> trackerPtr;
	Tparams params_;
	Rect2d boundingBox_;
};


#undef BLUR_AS_VADIM
#undef CLOSED_LOOP

class TrackerTLDImpl : public TrackerTLD
{
public:
	TrackerTLDImpl(const TrackerTLD::Params &parameters = TrackerTLD::Params());
	void read(const FileNode& fn) CV_OVERRIDE;
	void write(FileStorage& fs) const CV_OVERRIDE;

    Ptr<TrackerModel> getModel()
    {
      return model;
    }

	class Pexpert
	{
	public:
		Pexpert(const Mat& img_in, const Mat& imgBlurred_in, Rect2d& resultBox_in,
			const TLDDetector* detector_in, TrackerTLD::Params params_in, Size initSize_in) :
			img_(img_in), imgBlurred_(imgBlurred_in), resultBox_(resultBox_in), detector_(detector_in), params_(params_in), initSize_(initSize_in){}
		bool operator()(Rect2d /*box*/){ return false; }
		int additionalExamples(std::vector<Mat_<uchar> >& examplesForModel, std::vector<Mat_<uchar> >& examplesForEnsemble);
	protected:
		Pexpert() : detector_(NULL) {}
		Mat img_, imgBlurred_;
		Rect2d resultBox_;
		const TLDDetector* detector_;
		TrackerTLD::Params params_;
		RNG rng;
		Size initSize_;
	};

	class Nexpert : public Pexpert
	{
	public:
		Nexpert(const Mat& img_in, Rect2d& resultBox_in, const TLDDetector* detector_in, TrackerTLD::Params params_in)
		{
			img_ = img_in; resultBox_ = resultBox_in; detector_ = detector_in; params_ = params_in;
		}
		bool operator()(Rect2d box);
		int additionalExamples(std::vector<Mat_<uchar> >& examplesForModel, std::vector<Mat_<uchar> >& examplesForEnsemble)
		{
			examplesForModel.clear(); examplesForEnsemble.clear(); return 0;
		}
	};

	bool initImpl(const Mat& image, const Rect2d& boundingBox) CV_OVERRIDE;
	bool updateImpl(const Mat& image, Rect2d& boundingBox) CV_OVERRIDE;

	TrackerTLD::Params params;
	Ptr<Data> data;
	Ptr<TrackerProxy> trackerProxy;

};

}}}}  // namespace

#endif
