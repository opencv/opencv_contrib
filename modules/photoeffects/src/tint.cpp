#include "precomp.hpp"

using namespace cv;

namespace cv { namespace photoeffects {

void tint(InputArray src, OutputArray dst, const Vec3b &colorTint, float density)
{
    CV_Assert(src.type() == CV_8UC3);
    CV_Assert(density >= 0.0f && density <= 1.0f);
    dst.create(src.size(), CV_8UC3);
    Mat image = src.getMat(), outputImage = dst.getMat();
    Mat matColTint(src.size(), CV_8UC3,
        Scalar(colorTint[0], colorTint[1], colorTint[2]));

    outputImage = matColTint * density + image * (1 - density);
}

}}