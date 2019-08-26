
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ximgproc/edge_filter.hpp>

#include <string> 
#include <iostream>
#include <cstdlib>




#ifndef GLOBAL_MATTING_H
#define GLOBAL_MATTING_H

namespace cv
{
	
namespace ximgproc
{	
class GlobalMatting
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
	

	// for sorting the boundary pixels according to intensity
	struct IntensityComp
	{
	    IntensityComp(const cv::Mat_<cv::Vec3b> &image) : image(image)
	    {

	    }

	    bool operator()(const cv::Point &p0, const cv::Point &p1) const
	    {
		const cv::Vec3b &c0 = image(p0.y, p0.x);
		const cv::Vec3b &c1 = image(p1.y, p1.x);

		return ((int)c0[0] + (int)c0[1] + (int)c0[2]) < ((int)c1[0] + (int)c1[1] + (int)c1[2]);
	    }

	    const cv::Mat_<cv::Vec3b> &image;
	};

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
/*
int main(int argc,char** argv)
{
    if(argc<3)
    {
      cout<<"Enter the path of image and trimap"<<endl;
      return 0;
    }
    
    string img_path = argv[1];
    string tri_path = argv[2];
    int niter = 9;
    if(argc==4)
    {
      niter = atoi(argv[3]);
    } 
      
    cv::Mat image = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
    cv::Mat trimap = cv::imread(tri_path, CV_LOAD_IMAGE_GRAYSCALE);

    // (optional) exploit the affinity of neighboring pixels to reduce the 
    // size of the unknown region. please refer to the paper
    // 'Shared Sampling for Real-Time Alpha Matting'.
    expansionOfKnownRegions(image, trimap, niter);

    cv::Mat foreground, alpha;
    globalMatting(image, trimap, foreground, alpha);


    for (int x = 0; x < trimap.cols; ++x)
        for (int y = 0; y < trimap.rows; ++y)
        {
            if (trimap.at<uchar>(y, x) == 0)
                alpha.at<uchar>(y, x) = 0;
            else if (trimap.at<uchar>(y, x) == 255)
                alpha.at<uchar>(y, x) = 255;
        }

    cv::imwrite("GT04-alpha.png", alpha);

    return 0;
}
*/
