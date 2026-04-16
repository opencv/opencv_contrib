// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_AUG_RNG_HPP
#define OPENCV_AUG_RNG_HPP



namespace cv{

    namespace imgaug{
        //! @addtogroup imgaug
        //! @{

        //! Seed used to initialize cv::imgaug::rng. If you don't set it manually using cv::imgaug::setSeed,
        //! it is initialized from the current tick count returned by cv::getTickCount.
        extern uint64 state;

        //! Random number generator for the data augmentation module.
        extern cv::RNG rng;

        /** @brief Set the seed of cv::imgaug::rng.
         *
         * Re-seeding the generator makes imgaug operations reproducible: using the same seed, the same input,
         * and the same call order yields the same output. The random number generator state advances after each
         * call that consumes random numbers.
         *
         * This function does not guarantee bitwise-identical results across OpenCV versions or different
         * implementations.
         *
         * @param seed Seed value used to initialize cv::imgaug::rng.
         */
        CV_EXPORTS_W void setSeed(uint64 seed);

        //! @}
    }
}




#endif //OPENCV_AUG_RNG_HPP
