#include "precomp.hpp"
#include <vector>

using namespace std;

namespace cv { namespace photoeffects {

int sepia(InputArray src, OutputArray dst)
{
    CV_Assert(src.type() == CV_8UC1);
    Mat img = src.getMat(), hsvImg, sepiaH, sepiaS;
    Scalar hue(19), saturation(78), value(20);

    vector<Mat> sepiaPlanes;
    sepiaPlanes.resize(3);
    sepiaH.create(img.size(), CV_8UC1);
    sepiaS.create(img.size(), CV_8UC1);
    sepiaH.setTo(hue);
    sepiaS.setTo(saturation);
    sepiaPlanes[0] = sepiaH;
    sepiaPlanes[1] = sepiaS;
    sepiaPlanes[2] = img + value;
    merge(sepiaPlanes, hsvImg);

    cvtColor(hsvImg, dst, COLOR_HSV2BGR);
    return 0;
}

}}