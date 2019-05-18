// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_AUGMENT_FLIP_HPP
#define OPENCV_AUGMENT_FLIP_HPP
#include <opencv2/augment/Transform.hpp>

namespace cv {
namespace augment {

    //! @addtogroup augment
        //! @{

    class CV_EXPORTS_W FlipHorizontal : public Transform
    {
    public:
        /* @brief Constructor
            @param _probability probability that this transformation will be applied to each image
        */
        CV_WRAP FlipHorizontal(const Scalar& _probability);

        /* @brief Apply the horizontal flipping to a single image
            @param _src Input image to be flipped
            @param _dst Output (flipped) image
        */
        CV_WRAP void image(InputArray _src, OutputArray _dst);

        /* @brief Apply the flipping for a single point
            @param _src Input point to be flipped
        */
        virtual Point2d point(InputArray image, Point2d& src);

    };



    class CV_EXPORTS_W FlipVertical : public Transform
    {
    public:
        /* @brief Constructor
           @param _probability probability that this transformation will be applied to each image
        */
        CV_WRAP FlipVertical(const Scalar& _probability);

        /* @brief Apply the vertical flipping to a single image
           @param _src Input image to be flipped
           @param _dst Output (flipped) image
        */
        CV_WRAP void image(InputArray _src, OutputArray _dst);

        /* @brief Apply the flipping for a single point
           @param _src Input point to be flipped
        */
        virtual Point2d point(InputArray image, Point2d& src);

    };


    //! @}

} //augment
} //cv

#endif
