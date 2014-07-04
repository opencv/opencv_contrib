#include "BackgroundSubtractorLBSP.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iomanip>
#include <exception>

BackgroundSubtractorLBSP::BackgroundSubtractorLBSP(float fRelLBSPThreshold, size_t nDescDistThreshold, size_t nLBSPThresholdOffset)
	:	 nDebugCoordX(0),nDebugCoordY(0)
		,m_nKeyPoints(0)
		,m_nImgChannels(0)
		,m_nImgType(0)
		,m_nDescDistThreshold(nDescDistThreshold)
		,m_nLBSPThresholdOffset(nLBSPThresholdOffset)
		,m_fRelLBSPThreshold(fRelLBSPThreshold)
		,m_bInitialized(false) {
	CV_Assert(m_fRelLBSPThreshold>=0);
}

BackgroundSubtractorLBSP::~BackgroundSubtractorLBSP() {}

void BackgroundSubtractorLBSP::initialize(const cv::Mat& oInitImg) {
	this->initialize(oInitImg,std::vector<cv::KeyPoint>());
}

//cv::AlgorithmInfo* BackgroundSubtractorLBSP::info() const {
//	//return nullptr;
//}

void BackgroundSubtractorLBSP::getBackgroundDescriptorsImage(cv::OutputArray backgroundDescImage) const {
	CV_Assert(LBSP::DESC_SIZE==2);
	CV_Assert(m_bInitialized);
	cv::Mat oAvgBGDesc = cv::Mat::zeros(m_oImgSize,CV_32FC((int)m_nImgChannels));
	for(size_t n=0; n<m_voBGDescSamples.size(); ++n) {
		for(int y=0; y<m_oImgSize.height; ++y) {
			for(int x=0; x<m_oImgSize.width; ++x) {
				const size_t idx_ndesc = m_voBGDescSamples[n].step.p[0]*y + m_voBGDescSamples[n].step.p[1]*x;
				const size_t idx_flt32 = idx_ndesc*2;
				float* oAvgBgDescPtr = (float*)(oAvgBGDesc.data+idx_flt32);
				const ushort* const oBGDescPtr = (ushort*)(m_voBGDescSamples[n].data+idx_ndesc);
				for(size_t c=0; c<m_nImgChannels; ++c)
					oAvgBgDescPtr[c] += ((float)oBGDescPtr[c])/m_voBGDescSamples.size();
			}
		}
	}
	oAvgBGDesc.convertTo(backgroundDescImage,CV_16U);
}

std::vector<cv::KeyPoint> BackgroundSubtractorLBSP::getBGKeyPoints() const {
	return m_voKeyPoints;
}

void BackgroundSubtractorLBSP::setBGKeyPoints(std::vector<cv::KeyPoint>& keypoints) {
	LBSP::validateKeyPoints(keypoints,m_oImgSize);
	CV_Assert(!keypoints.empty());
	m_voKeyPoints = keypoints;
	m_nKeyPoints = keypoints.size();
}
