#include "precomp.hpp"
#include "dtfilter_cpu.hpp"
#include "dtfilter_ocl.hpp"

namespace cv
{

static bool dtUseOpenCLVersion(InputArray guide, InputArray src, int mode, int numIters)
{
    return 
        false && guide.isUMat() && ocl::useOpenCL() &&
        (guide.cols() >= 256 && guide.rows() >= 256) &&
        (guide.depth() == CV_32F || guide.depth() == CV_8U) &&
        (src.depth() == CV_32F || src.depth() == CV_8U) &&
        (numIters <= 3);
}

CV_EXPORTS_W
Ptr<DTFilter> createDTFilter(InputArray guide, double sigmaSpatial, double sigmaColor, int mode, int numIters)
{
    return Ptr<DTFilter>(DTFilterCPU::create(guide, sigmaSpatial, sigmaColor, mode, numIters));
}

CV_EXPORTS_W
void dtFilter(InputArray guide, InputArray src, OutputArray dst, double sigmaSpatial, double sigmaColor, int mode, int numIters)
{
    if (dtUseOpenCLVersion(guide, src, mode, numIters))
    {
        Ptr<DTFilterOCL> dtf = DTFilterOCL::create(guide, sigmaSpatial, sigmaColor, mode, numIters);
        dtf->setSingleFilterCall(true);
        dtf->filter(src, dst);
    }
    else
    {
        Ptr<DTFilterCPU> dtf = DTFilterCPU::create(guide, sigmaSpatial, sigmaColor, mode, numIters);
        dtf->setSingleFilterCall(true);
        dtf->filter(src, dst);
    }
}

CV_EXPORTS_W
Ptr<DTFilter> createDTFilterOCL(InputArray guide, double sigmaSpatial, double sigmaColor, int mode, int numIters)
{
    Ptr<DTFilterOCL> dtf = DTFilterOCL::create(guide, sigmaSpatial, sigmaColor, mode, numIters);
    return Ptr<DTFilter>(dtf);
}

}