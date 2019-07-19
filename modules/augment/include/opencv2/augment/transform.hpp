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
       @param src Input image to be tranformed
       @param dst Output (transformed) image
    */
    CV_WRAP virtual void image(InputArray src, OutputArray dst);

    /* @brief Apply the transformation for a single point (this is overridden by transformations implementation)
       @param src Input point to be tranformed
    */
    virtual Point2f point(const Point2f& src);


    /* @brief Apply the transformation for a rectangle
       @param box Rect2f consisting of (x1, y1, w, h) corresponding to (top left point, size)
    */
    virtual Rect2f rectangle(const Rect2f& src);

    /* @brief Apply the transformation for array of points
       @param src Mat consisting of the points to be transformed (each row is a point (X, Y))
       @param dst Output Mat that has the points transformed
    */
    CV_WRAP virtual void points(InputArray src, OutputArray dst);

    /* @brief Apply the transformation for array of rectangles
       @param src Mat consisting of the rectangles to be transformed (each row is a rectangle (x1, y1, w, h))
       @param dst Output Mat that has the rectangles transformed
    */
    CV_WRAP virtual void rectangles(InputArray src, OutputArray dst);

    /* @brief Apply the transformation for array of polygons
       @param src vector of Mat consisting of the polygons to be transformed (each row of a polygon Mat is a vertix (X,Y))
       @param dst the vector of Mat containing the point after applying the transformation
    */
    CV_WRAP virtual void polygons(std::vector<Mat> src, OutputArrayOfArrays dst);

    /* @brief Apply the transformation to a single mask (this is overridden by transformations implementation)
       @param src Input image to be tranformed
       @param dst Output (transformed) image
    */
    CV_WRAP virtual void mask(InputArray src, OutputArray dst);

    /* @brief set the random variables in a transformation to be used consitently on the next data
    */
    CV_WRAP virtual void init(const Mat& srcImage);


protected:
    static RNG rng;
    int srcImageRows;
    int srcImageCols;
};

}} //namespacw cv::augment
#endif
