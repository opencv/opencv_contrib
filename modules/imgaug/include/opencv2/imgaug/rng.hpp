// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_AUG_RNG_HPP
#define OPENCV_AUG_RNG_HPP



namespace cv{

    namespace imgaug{
        //! @addtogroup imgaug
        //! @{

        //! Initial state of the random number generator cv::imgaug::rng. If you don't manually set it using cv::imgaug::setSeed,
        //! it will be set to the current tick count returned by cv::getTickCount.
        extern uint64 state;

        //! Random number generator for data augmentation module
        extern cv::RNG rng;

        /** @brief Manually set the initial state of the random number generator cv::imgaug::rng.
         *
         * @param seed The seed value needed to generate a random number.
         */
        CV_EXPORTS_W void setSeed(uint64 seed);

        //! @}
    }
}




#endif //OPENCV_AUG_RNG_HPP
