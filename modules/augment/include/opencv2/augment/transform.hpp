// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_AUGMENT_TRANSFORM_HPP
#define OPENCV_AUGMENT_TRANSFORM_HPP
#include <opencv2/core.hpp>
#include <vector>

namespace cv {
namespace augment {
    //! @addtogroup augment
        //! @{


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
           @param image the image that has the point to be transformed
           @param src Input point to be tranformed
        */
        virtual Point2f point(InputArray image,const Point2f& src);


        /* @brief Apply the transformation for a rectangle
           @param image the image that has the rectangle to be transformed
           @param box Vec4f consisting of (x1, y1, x2, y1) corresponding to (top left, bottom right)
       */
        virtual Vec4f rectangle(InputArray image, const Vec4f& src);


        /* @brief Apply the transformation for array of points 
           @param image the image that has the points to be transformed
           @param _src Mat consisting of the points to be transformed (each row is a point (X, Y))
           @param _dst Output Mat that has the points transformed 
        */
        CV_WRAP virtual void points(InputArray image, InputArray _src, OutputArray _dst);

        /* @brief Apply the transformation for array of rectangles
           @param image the image that has the rectangles to be transformed
           @param _src Mat consisting of the rectangles to be transformed (each row is a rectangle (x1, y1, x2, y2))
           @param _dst Output Mat that has the rectangles transformed
        */
        CV_WRAP virtual void rectangles(InputArray image, InputArray _src, OutputArray _dst);


        /* @brief Apply the transformation for array of polygons
           @param image the image that has the polygons to be transformed
           @param src vector of Mat consisting of the polygons to be transformed (each row of a polygon Mat is a vertix (X,Y))
        */
        CV_WRAP virtual std::vector<Mat> polygons(InputArray image, std::vector<Mat> src);


        /* @brief set the random variables in a transformation to be used consitently on the next data 
        */
        CV_WRAP virtual void resetRandom();

    };


    //! @}

} //augment
} //cv
#endif
