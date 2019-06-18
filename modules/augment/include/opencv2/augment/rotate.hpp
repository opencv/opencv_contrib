// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_AUGMENT_ROTATE_HPP
#define OPENCV_AUGMENT_ROTATE_HPP
#include <opencv2/augment/Transform.hpp>

namespace cv {
namespace augment {

    //! @addtogroup augment
        //! @{

    class CV_EXPORTS_W Rotate : public Transform
    {
    public:
        /* @brief Constructor
           @param _angleRange the range of angle values to rotate the image
        */
        CV_WRAP Rotate(Range _angleRange);

        /* @brief Constructor to initialize the rotation transformation with default angleRange
        */
        CV_WRAP Rotate();

        /* @brief Apply the rotation to a single image
            @param _src Input image to be flipped
            @param _dst Output (rotated) image
        */
        CV_WRAP void image(InputArray _src, OutputArray _dst);

        /* @brief Apply the rotation for a single point
            @param _src Input point to be rotated
        */
        Point2f point(InputArray image,const Point2f& src);

        /* @brief choose an angle from the specified range to apply in next transformations
        */
        void resetRandom();

    private :
        Range angleRange;
        int currentAngle;
        RNG rng;

    };

    //! @}

} //augment
} //cv

#endif
