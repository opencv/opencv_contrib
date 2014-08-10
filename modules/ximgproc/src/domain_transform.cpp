#include "precomp.hpp"
#include "dtfilter_cpu.hpp"

namespace cv
{

CV_EXPORTS_W
Ptr<DTFilter> createDTFilter(InputArray guide, double sigmaSpatial, double sigmaColor, int mode, int numIters)
{
    return Ptr<DTFilter>(DTFilterCPU::create(guide, sigmaSpatial, sigmaColor, mode, numIters));
}

CV_EXPORTS_W
void dtFilter(InputArray guide, InputArray src, OutputArray dst, double sigmaSpatial, double sigmaColor, int mode, int numIters)
{
    Ptr<DTFilterCPU> dtf = DTFilterCPU::create(guide, sigmaSpatial, sigmaColor, mode, numIters);
    dtf->setSingleFilterCall(true);
    dtf->filter(src, dst);
}

}