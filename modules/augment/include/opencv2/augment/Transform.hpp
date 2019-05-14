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
            @param _probability probability that this transformation will be applied to each image
        */
        CV_WRAP Transform(const Scalar& _proability);

        /* @brief Destructor
        */
        virtual ~Transform();

        /*@brief returns the transform probability
        */
        CV_WRAP Scalar getProbability();

        /*@brief change the transform probability
            @param _probability the new probability to be assigned to the transform
        */
        CV_WRAP void setProbability(Scalar& _probability);

        /* @brief Apply the transformation to a single image (this is overridden by transformations implementation)
            @param _src Input image to be tranformed
            @param _dst Output (transformed) image
        */
        CV_WRAP virtual void image(InputArray _src, OutputArray _dst);

        /* @brief Apply the transformation for a single point (this is overridden by transformations implementation)
           @param image the image that has the point to be transformed
           @param src Input point to be tranformed
        */
        CV_WRAP virtual Point2d point(InputArray image, Point2d& src);


        /* @brief Apply the transformation for a single point (this is overridden by transformations implementation)
           @param image the image that has the point to be transformed
           @param x1 Minimum x of the rect
           @param y1 Minimum y of the rect
           @param x2 Maximum x of the rect
           @param y2 Maximum y of the rect
       */
        CV_WRAP virtual Scalar rect(InputArray image, Scalar box);


    protected:
        Scalar probability;
    };


    //! @}

} //augment
} //cv
#endif
