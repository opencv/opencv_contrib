#pragma once

#include <opencv2/features2d.hpp>
#include <opencv2/video/background_segm.hpp>
#include "LBSP.h"

/*!
	Local Binary Similarity Pattern (LBSP) foreground-background segmentation algorithm (abstract version).

	For more details on the different parameters, see P.-L. St-Charles and G.-A. Bilodeau, "Improving Background
	Subtraction using Local Binary Similarity Patterns", in WACV 2014, or G.-A. Bilodeau et al, "Change Detection
	in Feature Space Using Local Binary Similarity Patterns", in CRV 2013.

	This algorithm is currently NOT thread-safe.
 */
class BackgroundSubtractorLBSP  /* : public cv::BackgroundSubtractor */ {
public:
	//! full constructor
	BackgroundSubtractorLBSP(float fRelLBSPThreshold, size_t nDescDistThreshold, size_t nLBSPThresholdOffset=0);
	//! default destructor
	virtual ~BackgroundSubtractorLBSP();
	//! (re)initiaization method; needs to be called before starting background subtraction
	virtual void initialize(const cv::Mat& oInitImg);
	//! (re)initiaization method; needs to be called before starting background subtraction (note: also reinitializes the keypoints vector)
	virtual void initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints)=0;
	//! primary model update function; the learning param is used to override the internal learning speed (ignored when <= 0)
	virtual void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRate=0)=0;
	//! unused, always returns nullptr
	virtual cv::AlgorithmInfo* info() const;
	//! returns a copy of the latest reconstructed background descriptors image
	virtual void getBackgroundDescriptorsImage(cv::OutputArray backgroundDescImage) const;
	//! returns the keypoints list used for descriptor extraction (note: by default, these are generated from the DenseFeatureDetector class, and the border points are removed)
	virtual std::vector<cv::KeyPoint> getBGKeyPoints() const;
	//! sets the keypoints to be used for descriptor extraction, effectively setting the BGModel ROI (note: this function will remove all border keypoints)
	virtual void setBGKeyPoints(std::vector<cv::KeyPoint>& keypoints);

	// ######## DEBUG PURPOSES ONLY ##########
	int nDebugCoordX, nDebugCoordY;

protected:
	//! background model descriptors samples (tied to m_voKeyPoints but shaped like the input frames)
	std::vector<cv::Mat> m_voBGDescSamples;
	//! background model keypoints used for LBSP descriptor extraction (specific to the input image size)
	std::vector<cv::KeyPoint> m_voKeyPoints;
	//! defines the current number of used keypoints (always tied to m_voKeyPoints)
	size_t m_nKeyPoints;
	//! input image size
	cv::Size m_oImgSize;
	//! input image channel size
	size_t m_nImgChannels;
	//! input image type
	int m_nImgType;
	//! absolute descriptor distance threshold
	const size_t m_nDescDistThreshold;
	//! LBSP internal threshold offset value -- used to reduce texture noise in dark regions
	const size_t m_nLBSPThresholdOffset;
	//! LBSP relative internal threshold (kept here since we don't keep an LBSP object)
	const float m_fRelLBSPThreshold;
	//! pre-allocated internal LBSP threshold values for all possible 8-bit intensity values
	size_t m_anLBSPThreshold_8bitLUT[UCHAR_MAX+1];
	//! defines whether or not the subtractor is fully initialized
	bool m_bInitialized;
};

