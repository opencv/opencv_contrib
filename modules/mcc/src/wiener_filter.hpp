/**
  *  @file wiener_filter
  *  @brief filter wiener for denoise
  *  @author: Pedro D. Marrero Fernandez
  *  @data: 17/05/2016
  */


#ifndef _WIENER_FILTER_H
#define _WIENER_FILTER_H

#include "core.hpp"
namespace cv{
namespace mcc{

	/// CWienerFilter
	/**  @brief wiener class filter for denoise
	  *  @author: Pedro D. Marrero Fernandez
	  *  @data: 17/05/2016
	  */
	class CWienerFilter
	{
	public:

		CWienerFilter();
		~CWienerFilter();

		/** cvWiener2
		  * @brief A Wiener 2D Filter implementation for OpenCV
		  * @author: Ray Juang / rayver{ _at_ } hkn{ / _dot_ / } berkeley(_dot_) edu
		  * @date : 12.1.2006
		  */
		void wiener2(cv::Mat & src, cv::Mat & dest, int szWindowX, int szWindowY);

	private:


		void cvWiener2(const Mat& srcMat, Mat& dstMat, int szWindowX, int szWindowY);


	};


}

}

#endif //_WIENER_FILTER_H
