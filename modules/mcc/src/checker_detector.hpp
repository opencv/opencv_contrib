#ifndef _MCC_CHECKER_DETECTOR_H
#define _MCC_CHECKER_DETECTOR_H

#include "opencv2/mcc/checker_detector.hpp"
#include "core.hpp"
#include "opencv2/mcc/checker_model.hpp"
#include "charts.hpp"


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <vector>

// #define _DEBUG
// #define SHOW_DEBUG_IMAGES

#include "debug.hpp"
#include <time.h>

namespace cv{
namespace mcc{



class CCheckerDetectorImpl: public CCheckerDetector
{

	typedef std::vector<cv::Point> PointsVector;
	typedef std::vector<PointsVector> ContoursVector;



public:

	CCheckerDetectorImpl(float minerr = 2.0, int nc = 1, int fsize = 2000);
	virtual ~CCheckerDetectorImpl();





	/// process
	/** \brief process image
	* \param[in] image in color space BGR
	* \return state
	*/
	bool process(const cv::Mat & image, const int chartType) CV_OVERRIDE;

	bool process(const cv::Mat & image, const int chartType, std::vector<cv::Rect> regionsOfInterest) CV_OVERRIDE;


	#ifdef _DEBUG
	bool process(const cv::Mat & image, const std::string &pathOut, const int chartType) CV_OVERRIDE ;
	#endif

	/// getColorChecker
	/** \brief get the best color checker
	* \param[out] checker
	*/
	void getBestColorChecker(Ptr<CChecker> &checker) CV_OVERRIDE { checker = m_checkers[0]; }

	/// getColorChecker
	/** \brief get list color checker
	* \param[out] checkers list checker color
	*/
	void getListColorChecker(std::vector< Ptr<CChecker> > &checkers)  CV_OVERRIDE { checkers = m_checkers; }



protected: // methods pipeline


	/// prepareImage
	/** \brief Prepare Image
	  * \param[in] bgrMat image in color space BGR
	  * \param[out] grayOut gray scale
	  * \param[out] bgrOut rescale image
	  * \param[out] aspOut aspect ratio
	  * \param[in] max_size rescale factor in max dim
	  */
	virtual void
	prepareImage(const cv::Mat& bgr, cv::Mat& grayOut, cv::Mat &bgrOut, float &aspOut, int max_size = 2000) const;

	/// performThreshold
	/** \brief Adaptative threshold
	  * \param[in] grayscaleImg gray scale image
	  * \param[in] thresholdImg binary image
	  * \param[in] wndx, wndy windows size
	  * \param[in] step
	  */
	virtual void
	performThreshold(const cv::Mat& grayscaleImg, cv::Mat& thresholdImg);

	/// findContours
	/** \brief find contour in the image
	* \param[in] srcImg binary imagen
	* \param[out] contours
	* \param[in] minContourPointsAllowed
	*/
	virtual void
	findContours(const cv::Mat &srcImg, ContoursVector &contours, int minContourPointsAllowed) const;

	/// findCandidates
	/** \brief find posibel candidates
	* \param[in] contours
	* \param[out] detectedCharts
	* \param[in] minContourLengthAllowed
	*/
	virtual void
	findCandidates(const ContoursVector &contours, std::vector< CChart >& detectedCharts, int minContourLengthAllowed	);

	/// clustersAnalysis
	/** \brief clusters charts analysis
	* \param[in] detectedCharts
	* \param[out] groups
	*/
	virtual void
	clustersAnalysis(const std::vector< CChart > &detectedCharts, std::vector<int> &groups );

	/// checkerRecognize
	/** \brief checker color recognition
	* \param[in] img
	* \param[in] detectedCharts
	* \param[in] G
	* \param[out] colorChartsOut
	*/
	virtual void
	checkerRecognize(const Mat &img, const std::vector<CChart>& detectedCharts, const std::vector<int> &G,
		const int chartType,std::vector< std::vector<cv::Point2f> > &colorChartsOut);

	/// checkerAnalysis
	/** \brief evaluate checker
	* \param[in] img
	* \param[in] img_org
	* \param[in] colorCharts
	* \param[out] checker
	* \param[in] asp
	*/
	virtual void
	checkerAnalysis(const cv::Mat &img, const cv::Mat &img_org,
		const int chartType, std::vector< std::vector< cv::Point2f > > colorCharts,
		std::vector<Ptr<CChecker> > &checkers, float asp = 1);



protected:

	std::vector<Ptr<CChecker>> m_checkers;
	int m_fact_size;
	unsigned int m_num_ch;
	float m_min_error;


private: // methods aux

	void get_subbox_chart_physical(
		const std::vector<cv::Point2f> &points,
		std::vector<cv::Point2f> &chartPhy,
		cv::Size &size
		);


	void reduce_array(
		const std::vector<float> &x,
		std::vector<float> &x_new,
		float tol
		);


	void transform_points_forward(
		const cv::Matx33f &T,
		const std::vector<cv::Point2f> &X,
		std::vector<cv::Point2f> &Xt
		);


	void transform_points_inverse(
		const cv::Matx33f &T,
		const std::vector<cv::Point2f> &X,
		std::vector<cv::Point2f> &Xt
		);



	void get_profile(
		const cv::Mat &img,
		const std::vector< cv::Point2f > &ibox,
		const int chartType,
		cv::Mat &charts_rgb,
		cv::Mat &charts_ycbcr
		);



	/* \brief cost function
	 *  e(p) = ||f(p)||^2 = \sum_k (mu_{k,p}*r_k')/||mu_{k,p}||||r_k|| + ...
	 *                   + \sum_k || \sigma_{k,p} ||^2
	 */

	float cost_function( const cv::Mat &img, const std::vector<cv::Point2f > &ibox , const int chartType);


};


}
}

#endif //_MCC_CHECKER_DETECTOR_H
