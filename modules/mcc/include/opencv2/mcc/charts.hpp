#ifndef _MCC_CHARTS_H
#define _MCC_CHARTS_H

#include "core.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


namespace cv{
namespace mcc {


	/** \brief Chart model
	  *
	  * .--------. <- px,py
	  * |		 | one chart for checker color
	  * |  RGB   |
	  * |  Lab   |
	  * |        |
	  * .--------.
	  *
	  * \autor Pedro Marrero Fern�ndez
	  */

	class CChart
	{

	public:

		CChart();
		~CChart();

		/**\brief set corners
		  *\param p[in] new corners
		  */
		void setCorners( std::vector<cv::Point2f> p );


	public:

		std::vector<cv::Point2f> corners;
		cv::Point2f center;
		float perimetro;
		float area;
		float large_side;

	};




	/** \brief Chart draw */
	class CChartDraw
	{
	public:

		/**\brief contructor */
		CChartDraw(CChart *pChart, cv::Mat *image );

		/**\brief draw the chart contour over the image */
		void drawContour(cv::Scalar color = CV_RGB(0, 250, 0)) const;

		/**\brief draw the chart center over the image */
		void drawCenter(cv::Scalar color = CV_RGB(0, 0, 255)) const;

	private:

		CChart *m_pChart;
		cv::Mat *m_image;

	};



}
}

#endif //_MCC_CHARTS_H
