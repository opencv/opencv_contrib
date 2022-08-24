//
// Created by Chuyang Zhao on 2022/8/17.
//

#ifndef OPENCV_AUG_RNG_HPP
#define OPENCV_AUG_RNG_HPP

namespace cv{
    namespace imgaug{
        extern uint64 state;
        extern cv::RNG rng;

        CV_EXPORTS_W void setSeed(uint64 seed);
    }
}


#endif //OPENCV_AUG_RNG_HPP
