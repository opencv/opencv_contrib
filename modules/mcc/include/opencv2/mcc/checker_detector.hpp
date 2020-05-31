#ifndef __OPENCV_MCC_CHECKER_DETECTOR_HPP__
#define __OPENCV_MCC_CHECKER_DETECTOR_HPP__
#include <opencv2/core.hpp>
#include "checker_model.hpp"
#define _DEBUG
#define SHOW_DEBUG_IMAGES
namespace cv{
namespace mcc{
class CV_EXPORTS_W CCheckerDetector: public Algorithm
{
public:


	/// process
	/** \brief process image
	* \param image image in color space BGR
	* \param chartType type of the chart to detect
	* \return state
	*/
	virtual bool process(const cv::Mat & image, const int chartType)=0;

	#ifdef _DEBUG
	virtual bool process(const cv::Mat & image, const std::string &pathOut, const int chartType)=0;
	#endif

	/// getColorChecker
	/** \brief get the best color checker
	* \param[out] checker
	*/
	virtual void getBestColorChecker(Ptr<CChecker> &checker) =0;

	/// getColorChecker
	/** \brief get list color checker
	* \param[out] checkers list checker color
	*/
	virtual void getListColorChecker(std::vector< Ptr<CChecker> > &checkers)=0;


    static Ptr<CCheckerDetector> create(float minerr = 2.0, int nc = 1, int fsize = 2000);


};


}
}

#endif //_MCC_CHECKER_DETECTOR_H
