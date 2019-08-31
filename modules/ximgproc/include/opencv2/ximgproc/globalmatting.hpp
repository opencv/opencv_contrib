#ifndef GLOBAL_MATTING_H
#define GLOBAL_MATTING_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>

#include <string>
#include <cstdlib>




namespace cv
{

namespace ximgproc
{
class CV_EXPORTS GlobalMatting
{
      private:

	template <typename T>
        inline T sqr(T a)
        {
           return a * a;
        }



	std::vector<cv::Point> findBoundaryPixels(const cv::Mat_<uchar> &trimap, int a, int b);

	// Eq. 2
	float calculateAlpha(const cv::Vec3b &F, const cv::Vec3b &B, const cv::Vec3b &I);

	// Eq. 3
	float colorCost(const cv::Vec3b &F, const cv::Vec3b &B, const cv::Vec3b &I, float alpha);

	// Eq. 4
	float distCost(const cv::Point &p0, const cv::Point &p1, float minDist);

	float colorDist(const cv::Vec3b &I0, const cv::Vec3b &I1);
	float nearestDistance(const std::vector<cv::Point> &boundary, const cv::Point &p);




	void expansionOfKnownRegions(const cv::Mat_<cv::Vec3b> &image,
		                            cv::Mat_<uchar> &trimap,
                               int r, float c);

	// erode foreground and background regions to increase the size of unknown region
	void erodeFB(cv::Mat_<uchar> &trimap, int r);



	struct Sample
	{
	    int fi, bj;
	    float df, db;
	    float cost, alpha;
	};

	void calculateAlphaPatchMatch(const cv::Mat_<cv::Vec3b> &image,
		const cv::Mat_<uchar> &trimap,
		const std::vector<cv::Point> &foregroundBoundary,
		const std::vector<cv::Point> &backgroundBoundary,
		std::vector<std::vector<Sample> > &samples);

	void expansionOfKnownRegionsHelper(const cv::Mat &_image,
		                                  cv::Mat &_trimap,
		                                  int r, float c);


	// erode foreground and background regions to increase the size of unknown region
	void erodeFB(cv::Mat &_trimap, int r);

	void expansionOfKnownRegions(cv::InputArray _img, cv::InputOutputArray _trimap, int niter);
	void globalMattingHelper(cv::Mat _image, cv::Mat _trimap, cv::Mat &_foreground, cv::Mat &_alpha, cv::Mat &_conf);
        public:
                GlobalMatting();

		void globalMatting(cv::InputArray _image, cv::InputArray _trimap, cv::OutputArray _foreground, cv::OutputArray _alpha, cv::OutputArray _conf);

                void getMat(cv::Mat image,cv::Mat trimap,cv::Mat &foreground,cv:: Mat &alpha,int niter=9);

};

}
}
#endif
