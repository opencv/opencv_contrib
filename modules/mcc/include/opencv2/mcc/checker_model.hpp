
#ifndef __OPENCV_MCC_CHECKER_MODEL_HPP__
#define __OPENCV_MCC_CHECKER_MODEL_HPP__
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
namespace cv{
namespace mcc{

	class CV_EXPORTS_W CChecker
	{
    public:
		typedef
		enum _TYPECHAR
		{
			MCC24 = 0,
			SG140,
			PASSPORT

		}TYPECHRT;
	public:

		virtual ~CChecker() {}

        CV_WRAP static Ptr<CChecker> create();
	public:
		TYPECHRT target;				// o tipo de checkercolor
		int N;							// number of charts
		std::vector< cv::Point2f > box;	// corner box
		cv::Mat charts_rgb;				// charts profile rgb color space
		cv::Mat charts_ycbcr;			// charts profile YCbCr color space
		float cost;						// cost to aproximate
		cv::Point2f center;


	};
/** \brief checker draw
	  * \autor Pedro Marrero Fernandez
	  */
	class CV_EXPORTS_W CCheckerDraw
	{

	public:

        virtual ~CCheckerDraw(){}
		virtual void draw(cv::Mat &img, const int chartType)=0;
        static Ptr<CCheckerDraw> create(Ptr<CChecker> pChecker, cv::Scalar color= CV_RGB(0,250,0), int thickness=2);


	};

}
}

#endif
