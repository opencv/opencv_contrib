#ifndef _MCC_CHECKER_DETECTOR_H
#define _MCC_CHECKER_DETECTOR_H


#include "core.hpp"
#include "charts.hpp"
#include "checker_model.hpp"


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <vector>

#define _DEBUG
#define SHOW_DEBUG_IMAGES

#include "debug.hpp"
#include <time.h>


namespace cv{
namespace mcc{


class CChart;
class CChartClassicModel;


class CCheckerDetector
{

	typedef std::vector<cv::Point> PointsVector;
	typedef std::vector<PointsVector> ContoursVector;



public:

	CCheckerDetector(float minerr = 2.0, int nc = 1, int fsize = 1000);
	virtual ~CCheckerDetector();


	/// process
	/** \brief process image
	* \param[in] path image
	* \return state
	*/
	bool process(const std::string& pathImage);


	/// process
	/** \brief process image
	* \param[in] image in color space BGR
	* \return state
	*/
	bool process(const cv::Mat & image);

	#ifdef _DEBUG
	bool process(const cv::Mat & image, const std::string &pathOut);
	#endif

	/// getColorChecker
	/** \brief get the best color checker
	* \param[out] checker
	*/
	void getBestColorChecker(CChecker &checker) { checker = m_checkers[0]; }

	/// getColorChecker
	/** \brief get list color checker
	* \param[out] checkers list checker color
	*/
	void getListColorChecker(std::vector< CChecker > &checkers) { checkers = m_checkers; }

	/// startTracking
	/** \brief start tracking
	* \param[in] frame in color space BGR
	* \param[out] colorCharts
	* \return state
	*/
	bool startTracking(const cv::Mat & image, std::vector< std::vector<cv::Point2f > > &colorCharts);

	/// continueTracking
	/** \brief continue tracking
	* \param[in] frame in color space BGR
	* \param[out] colorCharts
	* \return state
	*/
	bool continueTracking(const cv::Mat & image, const std::vector< std::vector<cv::Point2f > > &colorCharts);



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
	prepareImage(const cv::Mat& bgr, cv::Mat& grayOut, cv::Mat &bgrOut, float &aspOut, int max_size = 1000) const;

	/// performThreshold
	/** \brief Adaptative threshold
	  * \param[in] grayscaleImg gray scale image
	  * \param[in] thresholdImg binary image
	  * \param[in] wndx, wndy windows size
	  * \param[in] step
	  */
	virtual void
	performThreshold(const cv::Mat& grayscaleImg, cv::Mat& thresholdImg, int wndx, int wndy, int step);

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
		std::vector< std::vector<cv::Point2f> > &colorChartsOut);

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
		std::vector< std::vector< cv::Point2f > > colorCharts,
		std::vector<CChecker>  &checkers, float asp = 1);



protected:

	std::vector<CChecker> m_checkers;
	int m_fact_size;
	int m_num_ch;
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
		cv::Mat &charts_rgb,
		cv::Mat &charts_ycbcr
		)
	{

		// color chart classic model
		CChartClassicModel cccm;
		cv::Mat lab; float J = 0; int N;
		std::vector<cv::Point2f>  fbox = cccm.box;
		std::vector<cv::Point2f> cellchart = cccm.cellchart;
		cv::Mat3f im_rgb, im_ycbcr, im_bgr(img);
		cv::Mat rgb[3], ycbcr[3];

		// Convert to RGB and YCbCr space
		cv::cvtColor(im_bgr, im_rgb, COLOR_BGR2RGB);
		cv::cvtColor(im_bgr, im_ycbcr, COLOR_BGR2YCrCb);

		// Get chanels
		split(im_rgb, rgb);
		split(im_ycbcr, ycbcr);


		// tranformation
		Matx33f ccT = cv::getPerspectiveTransform(fbox, ibox);

		cv::Mat mask;
		std::vector<cv::Point2f> bch(4), bcht(4);
		N = cellchart.size() / 4;


		// Create table charts information
		//		  |p_size|average|stddev|max|min|
		//	RGB   |      |       |      |   |   |
		//  YCbCr |

		charts_rgb = cv::Mat(cv::Size(5, 3*N), CV_32F);
		charts_ycbcr = cv::Mat(cv::Size(5, 3*N), CV_32F);

		cv::Scalar mu_rgb, st_rgb, mu_ycb, st_ycb, p_size;
		double 	max_rgb[3], min_rgb[3], max_ycb[3], min_ycb[3];


		for (size_t i = 0, k; i < N; i++)
		{
			k = 4 * i;
			bch[0] = cellchart[k + 0];
			bch[1] = cellchart[k + 1];
			bch[2] = cellchart[k + 2];
			bch[3] = cellchart[k + 3];
			polyanticlockwise(bch);
			transform_points_forward(ccT, bch, bcht);

			cv::Point2f c(0, 0);
			for (size_t j = 0; j < 4; j++) 	c += bcht[j];  c /= 4;
			for (size_t j = 0; j < 4; j++) 	bcht[j] = ((bcht[j] - c)*0.50) + c;


			mask = poly2mask(bcht, img.size());
			p_size = cv::sum(mask);

			// rgb space
			cv::meanStdDev(im_rgb, mu_rgb, st_rgb, mask);
			cv::minMaxLoc(rgb[0], &min_rgb[0], &max_rgb[0], NULL, NULL, mask);
			cv::minMaxLoc(rgb[1], &min_rgb[1], &max_rgb[1], NULL, NULL, mask);
			cv::minMaxLoc(rgb[2], &min_rgb[2], &max_rgb[2], NULL, NULL, mask);

			// create tabla
			//|p_size|average|stddev|max|min|
			// raw_r
			charts_rgb.at<float>(3 * i + 0, 0) = p_size(0);
			charts_rgb.at<float>(3 * i + 0, 1) = mu_rgb(0);
			charts_rgb.at<float>(3 * i + 0, 2) = st_rgb(0);
			charts_rgb.at<float>(3 * i + 0, 3) = min_rgb[0];
			charts_rgb.at<float>(3 * i + 0, 4) = max_rgb[0];
			// raw_g
			charts_rgb.at<float>(3 * i + 1, 0) = p_size(0);
			charts_rgb.at<float>(3 * i + 1, 1) = mu_rgb(1);
			charts_rgb.at<float>(3 * i + 1, 2) = st_rgb(1);
			charts_rgb.at<float>(3 * i + 1, 3) = min_rgb[1];
			charts_rgb.at<float>(3 * i + 1, 4) = max_rgb[1];
			// raw_b
			charts_rgb.at<float>(3 * i + 2, 0) = p_size(0);
			charts_rgb.at<float>(3 * i + 2, 1) = mu_rgb(2);
			charts_rgb.at<float>(3 * i + 2, 2) = st_rgb(2);
			charts_rgb.at<float>(3 * i + 2, 3) = min_rgb[2];
			charts_rgb.at<float>(3 * i + 2, 4) = max_rgb[2];


			// YCbCr space
			cv::meanStdDev(im_ycbcr, mu_ycb, st_ycb, mask);
			cv::minMaxLoc(ycbcr[0], &min_ycb[0], &max_ycb[0], NULL, NULL, mask);
			cv::minMaxLoc(ycbcr[1], &min_ycb[1], &max_ycb[1], NULL, NULL, mask);
			cv::minMaxLoc(ycbcr[2], &min_ycb[2], &max_ycb[2], NULL, NULL, mask);

			// create tabla
			//|p_size|average|stddev|max|min|
			// raw_Y
			charts_ycbcr.at<float>(3 * i + 0, 0) = p_size(0);
			charts_ycbcr.at<float>(3 * i + 0, 1) = mu_ycb(0);
			charts_ycbcr.at<float>(3 * i + 0, 2) = st_ycb(0);
			charts_ycbcr.at<float>(3 * i + 0, 3) = min_ycb[0];
			charts_ycbcr.at<float>(3 * i + 0, 4) = max_ycb[0];
			// raw_Cb
			charts_ycbcr.at<float>(3 * i + 1, 0) = p_size(0);
			charts_ycbcr.at<float>(3 * i + 1, 1) = mu_ycb(1);
			charts_ycbcr.at<float>(3 * i + 1, 2) = st_ycb(1);
			charts_ycbcr.at<float>(3 * i + 1, 3) = min_ycb[1];
			charts_ycbcr.at<float>(3 * i + 1, 4) = max_ycb[1];
			// raw_Cr
			charts_ycbcr.at<float>(3 * i + 2, 0) = p_size(0);
			charts_ycbcr.at<float>(3 * i + 2, 1) = mu_ycb(2);
			charts_ycbcr.at<float>(3 * i + 2, 2) = st_ycb(2);
			charts_ycbcr.at<float>(3 * i + 2, 3) = min_ycb[2];
			charts_ycbcr.at<float>(3 * i + 2, 4) = max_ycb[2];


		}
	}




	/* \brief cost function
	 *  e(p) = ||f(p)||^2 = \sum_k (mu_{k,p}*r_k')/||mu_{k,p}||||r_k|| + ...
	 *                   + \sum_k || \sigma_{k,p} ||^2
	 */

	float cost_function( const cv::Mat &img, const std::vector<cv::Point2f > &ibox );


};

}
}

#endif //_MCC_CHECKER_DETECTOR_H
