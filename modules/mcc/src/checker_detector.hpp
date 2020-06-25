#ifndef _MCC_CHECKER_DETECTOR_HPP
#define _MCC_CHECKER_DETECTOR_HPP

#include "opencv2/mcc/checker_detector.hpp"
#include "precomp.hpp"
#include "opencv2/mcc/checker_model.hpp"
#include "charts.hpp"


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <vector>



#include "debug.hpp"
#include <time.h>

namespace cv{
namespace mcc{



class CCheckerDetectorImpl: public CCheckerDetector
{

	typedef std::vector<cv::Point> PointsVector;
	typedef std::vector<PointsVector> ContoursVector;



public:

	CCheckerDetectorImpl();
	virtual ~CCheckerDetectorImpl();



	bool setNet(cv::dnn::Net _net) CV_OVERRIDE;

	/// process
	/** \brief process image
	* \param[in] image in color space BGR
	* \return state
	*/

	bool process(const cv::Mat & image, const TYPECHART chartType,const int nc = 1,
				bool use_net=false,
	             const Ptr<DetectorParameters> &params = DetectorParameters::create(),
				 std::vector<cv::Rect> regionsOfInterest = std::vector<cv::Rect>()
				 ) CV_OVERRIDE;


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


	bool _no_net_process(const cv::Mat & image, const TYPECHART chartType,
						 const int nc,
	                     const Ptr<DetectorParameters> &params,
				        std::vector<cv::Rect> regionsOfInterest);
	/// prepareImage
	/** \brief Prepare Image
	  * \param[in] bgrMat image in color space BGR
	  * \param[out] grayOut gray scale
	  * \param[out] bgrOut rescale image
	  * \param[out] aspOut aspect ratio
	  * \param[in] max_size rescale factor in max dim
	  */
	virtual void
	prepareImage(const cv::Mat& bgr, cv::Mat& grayOut, cv::Mat &bgrOut, float &aspOut,const Ptr<DetectorParameters> &params) const;

	/// performThreshold
	/** \brief Adaptative threshold
	  * \param[in] grayscaleImg gray scale image
	  * \param[in] thresholdImg binary image
	  * \param[in] wndx, wndy windows size
	  * \param[in] step
	  */
	virtual void
	performThreshold(const cv::Mat& grayscaleImg, std::vector<cv::Mat>& thresholdImg,const Ptr<DetectorParameters> &params) const;

	/// findContours
	/** \brief find contour in the image
	* \param[in] srcImg binary imagen
	* \param[out] contours
	* \param[in] minContourPointsAllowed
	*/
	virtual void
	findContours(const cv::Mat &srcImg, ContoursVector &contours, const Ptr<DetectorParameters> &params) const;

	/// findCandidates
	/** \brief find posibel candidates
	* \param[in] contours
	* \param[out] detectedCharts
	* \param[in] minContourLengthAllowed
	*/
	virtual void
	findCandidates(const ContoursVector &contours, std::vector< CChart >& detectedCharts, 	const Ptr<DetectorParameters> &params);

	/// clustersAnalysis
	/** \brief clusters charts analysis
	* \param[in] detectedCharts
	* \param[out] groups
	*/
	virtual void
	clustersAnalysis(const std::vector< CChart > &detectedCharts, std::vector<int> &groups, const Ptr<DetectorParameters> &params );

	/// checkerRecognize
	/** \brief checker color recognition
	* \param[in] img
	* \param[in] detectedCharts
	* \param[in] G
	* \param[out] colorChartsOut
	*/
	virtual void
	checkerRecognize(const Mat &img, const std::vector<CChart>& detectedCharts, const std::vector<int> &G,
		const TYPECHART chartType,std::vector< std::vector<cv::Point2f> > &colorChartsOut,
		const Ptr<DetectorParameters> &params);

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
		const TYPECHART chartType, const unsigned int nc,
		std::vector< std::vector< cv::Point2f > > colorCharts,
		std::vector<Ptr<CChecker> > &checkers, float asp,
		const Ptr<DetectorParameters>& params);

	virtual void
	removeTooCloseDetections(const Ptr<DetectorParameters> &params);


protected:

	std::vector<Ptr<CChecker>> m_checkers;
	cv::dnn::Net net;
	bool net_used = false;
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
		const TYPECHART chartType,
		cv::Mat &charts_rgb,
		cv::Mat &charts_ycbcr
		);



	/* \brief cost function
	 *  e(p) = ||f(p)||^2 = \sum_k (mu_{k,p}*r_k')/||mu_{k,p}||||r_k|| + ...
	 *                   + \sum_k || \sigma_{k,p} ||^2
	 */

	float cost_function( const cv::Mat &img, const std::vector<cv::Point2f > &ibox , const TYPECHART chartType);


};


}
}

#endif //_MCC_CHECKER_DETECTOR_HPP
