// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_AUGMENT_TRANSFORM_HPP
#define OPENCV_AUGMENT_TRANSFORM_HPP
#include <opencv2/core.hpp>
#include <vector>

namespace cv { namespace augment {

class CV_EXPORTS_W Transform
{
public:
    /* @brief Constructor
    */
    CV_WRAP Transform();

    /* @brief Destructor
    */
    virtual ~Transform();

    /* @brief Apply the transformation to a single image (this is overridden by transformations implementation)
        @param _src Input image to be tranformed
        @param _dst Output (transformed) image
    */
    CV_WRAP virtual void image(InputArray _src, OutputArray _dst);

    /* @brief Apply the transformation for a single point (this is overridden by transformations implementation)
        @param src Input point to be tranformed
    */
    virtual Point2f point(const Point2f& src);


    /* @brief Apply the transformation for a rectangle
        @param box Vec4f consisting of (x1, y1, x2, y1) corresponding to (top left, bottom right)
    */
    virtual Vec4f rectangle(const Vec4f& src);


    /* @brief Apply the transformation for array of points 
        @param _src Mat consisting of the points to be transformed (each row is a point (X, Y))
        @param _dst Output Mat that has the points transformed 
    */
    CV_WRAP virtual void points(InputArray _src, OutputArray _dst);

    /* @brief Apply the transformation for array of rectangles
        @param _src Mat consisting of the rectangles to be transformed (each row is a rectangle (x1, y1, x2, y2))
        @param _dst Output Mat that has the rectangles transformed
    */
    CV_WRAP virtual void rectangles(InputArray _src, OutputArray _dst);


    /* @brief Apply the transformation for array of polygons
        @param src vector of Mat consisting of the polygons to be transformed (each row of a polygon Mat is a vertix (X,Y))
    */
    CV_WRAP virtual std::vector<Mat> polygons(std::vector<Mat> src);


    /* @brief set the random variables in a transformation to be used consitently on the next data 
    */
    CV_WRAP virtual void init(Mat srcImage);


protected:
    static RNG rng;
    size_t srcImageRows;
    size_t srcImageCols;
};

}} //namespacw cv::augment
#endif
