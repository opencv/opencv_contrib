#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "DistanceUtils.h"

/*!
	Local Binary Similarity Pattern (LBSP) feature extractor

	Note 1: both grayscale and RGB/BGR images may be used with this extractor.
	Note 2: using LBSP::compute2(...) is logically equivalent to using LBSP::compute(...) followed by LBSP::reshapeDesc(...).

	For more details on the different parameters, see G.-A. Bilodeau et al, "Change Detection in Feature Space Using Local
	Binary Similarity Patterns", in CRV 2013.

	This algorithm is currently NOT thread-safe.
 */
class LBSP : public cv::DescriptorExtractor {
public:
	//! constructor 1, threshold = absolute intensity 'similarity' threshold used when computing comparisons
	LBSP(size_t nThreshold);
	//! constructor 2, threshold = relative intensity 'similarity' threshold used when computing comparisons
	LBSP(float fRelThreshold, size_t nThresholdOffset=0);
	//! default destructor
	virtual ~LBSP();
	//! loads extractor params from the specified file node @@@@ not impl
	virtual void read(const cv::FileNode&);
	//! writes extractor params to the specified file storage @@@@ not impl
	virtual void write(cv::FileStorage&) const;
	//! sets the 'reference' image to be used for inter-frame comparisons (note: if no image is set or if the image is empty, the algorithm will default back to intra-frame comparisons)
	virtual void setReference(const cv::Mat&);
	//! returns the current descriptor size, in bytes
	virtual int descriptorSize() const;
	//! returns the current descriptor data type
	virtual int descriptorType() const;
	//! returns whether this extractor is using a relative threshold or not
	virtual bool isUsingRelThreshold() const;
	//! returns the current relative threshold used for comparisons (-1 = invalid/not used)
	virtual float getRelThreshold() const;
	//! returns the current absolute threshold used for comparisons (-1 = invalid/not used)
	virtual size_t getAbsThreshold() const;

	//! similar to DescriptorExtractor::compute(const cv::Mat& image, ...), but in this case, the descriptors matrix has the same shape as the input matrix (possibly slower, but the result can be displayed)
	void compute2(const cv::Mat& oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat& oDescriptors) const;
	//! batch version of LBSP::compute2(const cv::Mat& image, ...), also similar to DescriptorExtractor::compute(const std::vector<cv::Mat>& imageCollection, ...)
	void compute2(const std::vector<cv::Mat>& voImageCollection, std::vector<std::vector<cv::KeyPoint> >& vvoPointCollection, std::vector<cv::Mat>& voDescCollection) const;

	// utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (1-channel version)
	inline static void computeGrayscaleDescriptor(const cv::Mat& oInputImg, const uchar _ref, const int _x, const int _y, const size_t _t, ushort& _res) {
		CV_DbgAssert(!oInputImg.empty());
		CV_DbgAssert(oInputImg.type()==CV_8UC1);
		CV_DbgAssert(LBSP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(_x>=(int)LBSP::PATCH_SIZE/2 && _y>=(int)LBSP::PATCH_SIZE/2);
		CV_DbgAssert(_x<oInputImg.cols-(int)LBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LBSP::PATCH_SIZE/2);
		const size_t _step_row = oInputImg.step.p[0];
		const uchar* const _data = oInputImg.data;
		#include "LBSP_16bits_dbcross_1ch.i"
	}

	// utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (3-channels version)
	inline static void computeRGBDescriptor(const cv::Mat& oInputImg, const uchar* const _ref,  const int _x, const int _y, const size_t* const _t, ushort* _res) {
		CV_DbgAssert(!oInputImg.empty());
		CV_DbgAssert(oInputImg.type()==CV_8UC3);
		CV_DbgAssert(LBSP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(_x>=(int)LBSP::PATCH_SIZE/2 && _y>=(int)LBSP::PATCH_SIZE/2);
		CV_DbgAssert(_x<oInputImg.cols-(int)LBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LBSP::PATCH_SIZE/2);
		const size_t _step_row = oInputImg.step.p[0];
		const uchar* const _data = oInputImg.data;
		#include "LBSP_16bits_dbcross_3ch3t.i"
	}

	// utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (3-channels version)
	inline static void computeRGBDescriptor(const cv::Mat& oInputImg, const uchar* const _ref,  const int _x, const int _y, const size_t _t, ushort* _res) {
		CV_DbgAssert(!oInputImg.empty());
		CV_DbgAssert(oInputImg.type()==CV_8UC3);
		CV_DbgAssert(LBSP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(_x>=(int)LBSP::PATCH_SIZE/2 && _y>=(int)LBSP::PATCH_SIZE/2);
		CV_DbgAssert(_x<oInputImg.cols-(int)LBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LBSP::PATCH_SIZE/2);
		const size_t _step_row = oInputImg.step.p[0];
		const uchar* const _data = oInputImg.data;
		#include "LBSP_16bits_dbcross_3ch1t.i"
	}

	// utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (1-channel-RGB version)
	inline static void computeSingleRGBDescriptor(const cv::Mat& oInputImg, const uchar _ref, const int _x, const int _y, const size_t _c, const size_t _t, ushort& _res) {
		CV_DbgAssert(!oInputImg.empty());
		CV_DbgAssert(oInputImg.type()==CV_8UC3 && _c<3);
		CV_DbgAssert(LBSP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(_x>=(int)LBSP::PATCH_SIZE/2 && _y>=(int)LBSP::PATCH_SIZE/2);
		CV_DbgAssert(_x<oInputImg.cols-(int)LBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LBSP::PATCH_SIZE/2);
		const size_t _step_row = oInputImg.step.p[0];
		const uchar* const _data = oInputImg.data;
		#include "LBSP_16bits_dbcross_s3ch.i"
	}

	//! utility function, used to reshape a descriptors matrix to its input image size via their keypoint locations
	static void reshapeDesc(cv::Size oSize, const std::vector<cv::KeyPoint>& voKeypoints, const cv::Mat& oDescriptors, cv::Mat& oOutput);
	//! utility function, used to illustrate the difference between two descriptor images
	static void calcDescImgDiff(const cv::Mat& oDesc1, const cv::Mat& oDesc2, cv::Mat& oOutput, bool bForceMergeChannels=false);
	//! utility function, used to filter out bad keypoints that would trigger out of bounds error because they're too close to the image border
	static void validateKeyPoints(std::vector<cv::KeyPoint>& voKeypoints, cv::Size oImgSize);
	//! utility, specifies the pixel size of the pattern used (width and height)
	static const size_t PATCH_SIZE = 5;
	//! utility, specifies the number of bytes per descriptor (should be the same as calling 'descriptorSize()')
	static const size_t DESC_SIZE = 2;

protected:
	//! classic 'compute' implementation, based on the regular DescriptorExtractor::computeImpl arguments & expected output
	virtual void computeImpl(const cv::Mat& oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat& oDescriptors) const;

	const bool m_bOnlyUsingAbsThreshold;
	const float m_fRelThreshold;
	const size_t m_nThreshold;
	cv::Mat m_oRefImage;
};
