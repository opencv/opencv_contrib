// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_AUGMENT_RESIZE_HPP
#define OPENCV_AUGMENT_RESIZE_HPP
#include <opencv2/augment/transform.hpp>

namespace cv { namespace augment {

class CV_EXPORTS_W Resize : public Transform
{
public:
   
    CV_WRAP Resize(Size size, std::vector<int>& interpolations);


    CV_WRAP Resize(Size size, int interpolation);


    CV_WRAP Resize(float fx, float fy, std::vector<int>& interpolations);


    CV_WRAP Resize(float fx, float fy, int interpolation);

    
    CV_WRAP virtual void image(InputArray src, OutputArray dst);

   
    Point2f virtual point(const Point2f& src);

 
    void virtual init(const Mat& srcImage);

private :
    Size size;
    float fx, fy;
    bool useSize;
    std::vector<int> interpolations;
    int interpolation;

};

}} //namespace cv::augment

#endif
