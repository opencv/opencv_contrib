// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "precomp.hpp"
#include <utility>

namespace cv{
    namespace imgaug{
        uint64 state = getTickCount();
        RNG rng(state);

        RNG& getRNG()
        {
            return rng;
        }

        void shuffleIntArray(int* first, int* last)
        {
            RNG& rngRef = getRNG();
            for (int i = static_cast<int>(last - first) - 1; i > 0; --i)
            {
                std::swap(first[i], first[rngRef.uniform(0, i + 1)]);
            }
        }

        void setSeed(uint64 seed){
            state = seed;
            rng.state = seed;
        }
    }
}
