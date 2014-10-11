#include "precomp.hpp"

namespace cv { namespace photoeffects {

void filmGrain(InputArray src, OutputArray dst, int grainValue, RNG& rng)
{
    CV_Assert(!src.empty());
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3);

    Mat image=src.getMat();
    Mat noise;
    noise.create(image.size(), CV_8UC1);
    rng.fill(noise, RNG::UNIFORM, 0, grainValue);
    dst.create(src.size(), src.type());
    Mat dstMat=dst.getMat();
    if(src.type()==CV_8UC3)
    {
        cvtColor(noise, noise, COLOR_GRAY2RGB);
    }
    dstMat=image+noise;
}

}}
