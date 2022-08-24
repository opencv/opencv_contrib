#include "precomp.hpp"

namespace cv{
    namespace imgaug{
        uint64 state = getTickCount();
        RNG rng(state);

        void setSeed(uint64 seed){
            rng.state = seed;
        }
    }
}