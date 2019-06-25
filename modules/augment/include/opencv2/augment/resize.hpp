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
    /* @brief Constructor
       @param size the size the images will be resized to 
       @param interpolations vector of the interpolation techniques to be chosen at random
    */
    CV_WRAP Resize(Size size, std::vector<int>& interpolations);

    /* @brief Constructor
       @param size the size the images will be resized to
       @param interpolation specific interpolation type to be used
    */
    CV_WRAP Resize(Size size, int interpolation);

    /* @brief Constructor
      @param fx a horizontal scale to be scaled with 
      @param fy a vertical scale to be scaled with
      @param interpolations vector of the interpolation techniques to be chosen at random
    */
    CV_WRAP Resize(float fx, float fy, std::vector<int>& interpolations);

    /* @brief Constructor
     @param fx a horizontal scale to be scaled with
     @param fy a vertical scale to be scaled with
     @param interpolation specific interpolation type to be used
   */
    CV_WRAP Resize(float fx, float fy, int interpolation);

    /* @brief Apply the resizing to a single image
       @param src Input image to be resized
       @param dst Output (resized) image
    */
    CV_WRAP virtual void image(InputArray src, OutputArray dst);

    /* @brief Apply the rotation for a single point
         @param _src Input point to be rotated
    */
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
