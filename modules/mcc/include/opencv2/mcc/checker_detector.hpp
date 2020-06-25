#ifndef __OPENCV_MCC_CHECKER_DETECTOR_HPP__
#define __OPENCV_MCC_CHECKER_DETECTOR_HPP__
#include <opencv2/core.hpp>
#include "checker_model.hpp"
#include <opencv2/dnn.hpp>

//uncomment the following two lines to view debugging output
// #define _DEBUG
// #define SHOW_DEBUG_IMAGES

namespace cv{
namespace mcc{
using namespace dnn;
//! @addtogroup mcc
//! @{



/**
 * @brief Parameters for the detectMarker process:
 */
struct CV_EXPORTS_W DetectorParameters {

    DetectorParameters();

    CV_WRAP static Ptr<DetectorParameters> create();

    CV_PROP_RW int adaptiveThreshWinSizeMin; ///<minimum window size for adaptive thresholding before finding contours (default 83).
    CV_PROP_RW int adaptiveThreshWinSizeMax; ///<maximum window size for adaptive thresholding before finding contours (default 153).
    CV_PROP_RW int adaptiveThreshWinSizeStep;///<increments from adaptiveThreshWinSizeMin to adaptiveThreshWinSizeMax during the thresholding (default 10).
    CV_PROP_RW double adaptiveThreshConstant; ///<constant for adaptive thresholding before finding contours (default 0)
    CV_PROP_RW double minContoursAreaRate;///<determine minimum area for marker contour to be detected. This is defined as a rate respect to the area of the input image. Used only if neural network is used (default 0.003).
    CV_PROP_RW double minContoursArea;///<determine minimum area for marker contour to be detected. This is defined as the actual area. Used only if neural network is not used (default 200).
	CV_PROP_RW double confidenceThreshold;///<minimum confidence for a bounding box detected by neural network to classify as detection. (default 0.5) (0<=confidenceThreshold<=1)
	CV_PROP_RW double minContourSolidity;///<minimum solidity of a contour for it be detected as a square in the chart. (default 0.9).
	CV_PROP_RW double findCandidatesApproxPolyDPEpsMultiplier;///<multipler to be used in cv::ApproxPolyDP function (default 0.05)
	CV_PROP_RW int borderWidth;///<width of the padding used to pass the inital neural network detection in the succeeding system.(default 0)
	CV_PROP_RW float B0factor;///< distance between two neighbours squares of the same chart. Defined as the ratio between distance and large dimension of square (default 1.25)
	CV_PROP_RW float maxError;///<maximum allowed error in the detection of a chart. default(0.1)
	CV_PROP_RW int minContourPointsAllowed;///< minium points in a detected contour. default(4)
	CV_PROP_RW int minContourLengthAllowed;///<minimum length of a countour. default(100)
	CV_PROP_RW int minInterContourDistance;///< minimum distance between two contours. default(100)
	CV_PROP_RW int minInterCheckerDistance;///<minimum distance between two checkers. default(10000)
	CV_PROP_RW int minImageSize;///<minimum size of the smaller dimension of the image. default(2000)
	CV_PROP_RW unsigned minGroupSize;///<minimum number of a squared of a chart that must be detected. default(4)

};



/** @brief A class to find the positions of the ColorCharts in the image.
 */

class CV_EXPORTS_W CCheckerDetector: public Algorithm
{
public:

	/** \brief Set the net which will be used to find the approximate
	*         bounding boxes for the color charts.
	*
	* It is not necessary to use this, but this usually results in
    * better detection rate.
	*
	* \param net the neural network, if the network in empty, then
	*            the function will return false.
	* \return true if it was able to set the detector's network,
    *         false otherwise.
	*/


	CV_WRAP virtual bool setNet(Net net) = 0;

	/** \brief Find the ColorCharts in the given image.
	*
	* The found charts are not returned but instead stored in the
	* detector, these can be accessed later on using getBestColorChecker()
	* and getListColorChecker()
	* \param image image in color space BGR
	* \param chartType type of the chart to detect
	* \param nc number of charts in the image, if you don't know the exact
    *           then keeping this number high helps.
	* \param useNet if it is true the network provided using the setNet()
	*                is used for preliminary search for regions where chart
	*                could be present, inside the regionsOfInterest provied.
	* \param params parameters of the detection system. More information
    *               about them can be found in the struct DetectorParameters.
	* \param regionsOfInterest regions of image to look for the chart, if
	*                          it is empty, charts are looked for in the
	*						   entire image
	* \return true if atleast one chart is detected otherwise false
	*/

	CV_WRAP virtual bool
	process(const cv::Mat &image, const TYPECHART chartType, const int nc = 1,
			bool useNet=false,
			const Ptr<DetectorParameters> &params = DetectorParameters::create(),
			std::vector<Rect> regionsOfInterest = std::vector<cv::Rect>()
		    ) = 0 ;

	/** \brief Get the best color checker. By the best it means the one
	*         detected with the highest confidence.
	* \param checker [out] A single colorchecker
	*/
	CV_WRAP virtual void getBestColorChecker(Ptr<mcc::CChecker> &checker) = 0;

	/** \brief Get the list of all detected colorcheckers
	* \param checkers [out] vector of colorcheckers
	*/
	CV_WRAP virtual void getListColorChecker(std::vector<Ptr<CChecker>> &checkers) = 0;

	/** \brief Returns the implementation of the CCheckerDetector.
	*
    */
	CV_WRAP static Ptr<CCheckerDetector> create();
};

//! @} mcc

} // namespace mcc
} // namespace cv

#endif
