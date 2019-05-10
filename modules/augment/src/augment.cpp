#include "opencv2/augment.hpp"
#include <iostream>

namespace cv
{
    namespace augment
    {
	    CV_EXPORTS_W void data_augment()
        {
            std::cout << "data augmentation is ON" << std::endl;
        }
    }
}
