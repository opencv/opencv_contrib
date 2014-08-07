#include "precomp.hpp"
#include "dtfilter_ocl.hpp"
#include "modules/ximgproc/opencl_kernels.hpp"
//#include <opencv2/highgui.hpp>
using namespace std;

#define MEASURE_TIME(counter, code) { counter -= getTickCount(); {code;} counter += getTickCount(); }
#define MEASURE_KER_TIME(code) { kernelsTime -= getTickCount(); code; kernelsTime += getTickCount(); }

namespace cv
{
    
int64 DTFilterOCL::totalKernelsTime = 0;

Ptr<DTFilterOCL> DTFilterOCL::create(InputArray guide, double sigmaSpatial, double sigmaColor, int mode, int numIters)
{
    DTFilterOCL *p = new DTFilterOCL();
    p->init(guide, sigmaSpatial, sigmaColor, mode, numIters);
    return Ptr<DTFilterOCL>(p);
}

void DTFilterOCL::init(InputArray guide_, double sigmaSpatial_, double sigmaColor_, int mode_, int numIters_)
{
    CV_Assert( guide_.channels() <= 4 && (guide_.depth() == CV_8U || guide_.depth() == CV_32F) );
    CV_Assert( numIters_ <= 3);

    mode                = mode_;
    numIters            = numIters_;
    sigmaSpatial        = std::max(1.00f, (float)sigmaSpatial_);
    sigmaColor          = std::max(0.01f, (float)sigmaColor_);
    kernelsTime         = 0;
    singleFilterCall    = false;
    h                   = guide_.rows();
    w                   = guide_.cols();

    NC_ocl_implememtation = ALG_INDEX;

    UMat guide = guide_.getUMat();
    initProgram();
    setBuildOptionsDT(guide);

    if (mode == DTF_NC)
    {
        initDT_NC(guide);
    }
    else if (mode == DTF_RF)
    {
        initDT_RF(guide);
    }
    else if (mode == DTF_IC)
    {
        initDT_IC(guide);
    }
    else
    {
        CV_Error(Error::StsBadFlag, "Incorrect DT filter mode");
    }
}


void DTFilterOCL::initDT_NC(UMat &guide)
{
    bool useIDT = false;
    if (useIDT)
    {
        UMat guideT;
        MEASURE_KER_TIME( transpose(guide, guideT); );
        computeIDT(guide, guideT);

        if (NC_ocl_implememtation == ALG_INDEX)
        {
            computeIndexMat(idistHor, distIndexHor);
            computeIndexMat(idistVert, distIndexVert);
            idistHor.release();
            idistVert.release();
        }
    }
    else
    {
        UMat dsitHorWrapper, distVertWrapper;
        computeDT(guide, dsitHorWrapper, distVertWrapper);

        distHor = dsitHorWrapper(Range::all(), Range(1, w + 1));

        MEASURE_KER_TIME( transpose(distVertWrapper, distVertWrapper); );
        distVert = distVertWrapper(Range::all(), Range(1, h + 1));

        if (NC_ocl_implememtation == ALG_INDEX)
        {
            computeIndexMat(distHor, distIndexHor, false);
            computeIndexMat(distVert, distIndexVert, false);
            distHor.release();
            distVert.release();
        }
    }
}

void DTFilterOCL::initDT_RF(UMat &guide)
{
    float sigmaRatio = (float)(sigmaSpatial / sigmaColor);
    float alpha1 = getIterAlpha(1);

    ocl::Kernel kerHor("compute_a0DT_vert", kerProgSrcDT, buildOptionsDT);
    ocl::Kernel kerVert("compute_a0DT_vert", kerProgSrcDT, buildOptionsDT);
    CV_Assert(!kerHor.empty() && !kerVert.empty());

    UMat guideT;
    transpose(guide, guideT);
    a0distHor.create(guide.cols - 1, guide.rows, CV_32FC1);
    a0distVert.create(guide.rows - 1, guide.cols, CV_32FC1);

    kerHor.args(
        ocl::KernelArg::ReadOnly(guideT),
        ocl::KernelArg::ReadWriteNoSize(a0distHor),
        sigmaRatio, alpha1);

    kerVert.args(
        ocl::KernelArg::ReadOnly(guide),
        ocl::KernelArg::ReadWriteNoSize(a0distVert),
        sigmaRatio, alpha1);

    size_t globalSizeHor[] = { guide.cols - 1, guide.rows };
    size_t globalSizeVert[] = { guide.rows - 1, guide.cols };
    size_t localSizeHor[] = { 1, getMaxWorkGropSize() };
    size_t localSizeVert[] = { 1, getMaxWorkGropSize() };
    bool sync = true;

    MEASURE_KER_TIME(kerHor.run(2, globalSizeHor, localSizeHor, sync));
    MEASURE_KER_TIME(kerVert.run(2, globalSizeVert, localSizeVert, sync));
}

void DTFilterOCL::initDT_IC(UMat& guide)
{
    UMat distHorWrapper, distVertWrapper;
    computeDT(guide, distHorWrapper, distVertWrapper);

    distHor = distHorWrapper(Range::all(), Range(1, w + 1));
    distVert = distVertWrapper(Range(1, 1 + h), Range::all());

    UMat distHorWrapperT;
    MEASURE_KER_TIME( transpose(distHorWrapper, distHorWrapperT); );
    distHorT = distHorWrapperT(Range(1, w + 1), Range::all());

    UMat distVertWrapperT;
    MEASURE_KER_TIME( transpose(distVertWrapper, distVertWrapperT); );
    distVertT = distVertWrapperT(Range::all(), Range(1, h + 1));

    computeIndexAndTailMat(distHor, distIndexHor, tailHor);
    computeIndexAndTailMat(distVertT, distIndexVert, tailVert);
}


void DTFilterOCL::filter(InputArray src_, OutputArray dst_, int dDepth)
{
    CV_Assert( src_.channels() <= 4 && (src_.depth() == CV_8U || src_.depth() == CV_32F) );
    CV_Assert( src_.cols() == w && src_.rows() == h );

    setBuildOptionsFlt(src_);

    if (dDepth == -1) dDepth = src_.depth();

    if (mode == DTF_NC)
    {
        UMat src = UMatAligned(h, w, workType);
        UMat dst;
        MEASURE_KER_TIME( src_.getUMat().convertTo(src, workType) );

        if (NC_ocl_implememtation == ALG_PER_ROW)
        {
            filterNC_perRowAlg(src, dst);
        }
        else if (NC_ocl_implememtation == ALG_PER_PIXEL)
        {
            filterNC_PerPixelAlg(src, dst);
        }
        else if (NC_ocl_implememtation == ALG_INDEX)
        {
            filterNC_IndexAlg(src, dst);
        }
            
        dst.convertTo(dst_, dDepth);
    }
    else if (mode == DTF_RF)
    {
        filterRF(src_, dst_, dDepth);
    }
    else if (mode == DTF_IC)
    {
        filterIC(src_, dst_, dDepth);
    }
    else
    {
        CV_Error(Error::StsBadFlag, "Incorrect DT filter mode");
    }
}

void DTFilterOCL::initProgram()
{
    //CV_Assert(ocl::Device::getDefault().type() == ocl::Device::TYPE_GPU);

    kerProgSrcDT = ocl::ximgproc::dtfilter_dt_oclsrc;
    kerProgSrcFilter = ocl::ximgproc::dtfilter_flt_oclsrc;
}

DTFilterOCL::DTFilterOCL()
    : a0distHor(distHor), a0distVert(distVert)
{

}

DTFilterOCL::~DTFilterOCL()
{
    totalKernelsTime += kernelsTime;
    //std::cout << "Kernels time " << kernelsTime / getTickFrequency() << " sec.\n";
    //std::cout << "Total kernels time " << totalKernelsTime / getTickFrequency() << " sec.\n";
}

void DTFilterOCL::integrateCols(UMat& src, UMat& isrc)
{
    CV_Assert(src.depth() == CV_32F);
    int sizeof_float4 = 16;

    int reqChuncks4f = (src.cols*src.elemSize() + sizeof_float4 - 1) / sizeof_float4;
    int maxChuncks4f = src.step.p[0] / sizeof_float4;
    CV_Assert(reqChuncks4f <= maxChuncks4f);

    createAligned(isrc, src.rows + 1, src.cols, src.type(), sizeof_float4, USAGE_ALLOCATE_DEVICE_MEMORY);

    ocl::Kernel ker = ocl::Kernel("integrate_cols_4f", kerProgSrcFilter, buildOptionsFlt);
    CV_Assert(!ker.empty());

    ker.args(ocl::KernelArg::ReadOnlyNoSize(src), ocl::KernelArg::WriteOnlyNoSize(isrc), (int)src.rows, reqChuncks4f);
    size_t globKerSize4f[] = { reqChuncks4f };
    bool oclKerStatus;
    MEASURE_KER_TIME( oclKerStatus = ker.run(1, globKerSize4f, NULL, true); );
    CV_Assert(oclKerStatus);
}

void DTFilterOCL::computeIndexMat(UMat& _dist, UMat& distIndex, bool useIDT)
{
    distIndex.create(_dist.rows*numIters, 2*_dist.cols, CV_32SC1);

    const char *kernelName = (useIDT) ? "find_conv_bounds_by_idt" : "find_conv_bounds_by_dt";
    ocl::Kernel ker = ocl::Kernel(kernelName, kerProgSrcDT, buildOptionsDT);
    ker.args(
        ocl::KernelArg::ReadOnly(_dist),
        ocl::KernelArg::WriteOnlyNoSize(distIndex),
        getIterRadius(1), getIterRadius(2), getIterRadius(3)
        );

    size_t globSize[] = { _dist.rows, _dist.cols };
    size_t localSize[] = { 1, getMaxWorkGropSize() };
    bool oclKernelStatus;
    MEASURE_KER_TIME(oclKernelStatus = ker.run(2, globSize, localSize, true));

    CV_Assert(oclKernelStatus);
}

void DTFilterOCL::computeIndexAndTailMat(UMat& dist, UMat& distIndex, UMat& tail)
{
    distIndex.create(dist.rows*numIters, 2*dist.cols, CV_32SC1);
    tail.create(dist.rows*numIters, 2*dist.cols, CV_32FC1);

    ocl::Kernel ker = ocl::Kernel("find_conv_bounds_and_tails", kerProgSrcDT, buildOptionsDT);
    CV_Assert(!ker.empty());

    ker.args(
        ocl::KernelArg::ReadOnly(dist),
        ocl::KernelArg::WriteOnlyNoSize(distIndex),
        ocl::KernelArg::WriteOnlyNoSize(tail),
        getIterRadius(1), getIterRadius(2), getIterRadius(3)
        );

    size_t globSize[] = {dist.rows, dist.cols};
    size_t localSize[] = {1, getMaxWorkGropSize()};
    bool oclKerStatus;
    
    MEASURE_KER_TIME( oclKerStatus = ker.run(2, globSize, localSize, true); );
    CV_Assert(oclKerStatus);
}

int DTFilterOCL::getMaxWorkGropSize()
{
    int maxWorkGropSize = ocl::Device::getDefault().maxWorkGroupSize();
    return (maxWorkGropSize <= 0) ? 256 : maxWorkGropSize;
}

int DTFilterOCL::getCacheLineSize()
{
    int lineSize = ocl::Device::getDefault().globalMemCacheLineSize();
    return (lineSize <= 0) ? 128 : lineSize;
}

void DTFilterOCL::filterRF(InputArray src_, OutputArray dst_, int dDepth)
{
    UMat res = UMatAligned(h, w, workType, getCacheLineSize());
    UMat resT = UMatAligned(w, h, workType, getCacheLineSize());
    UMat adistHor, adistVert;
    MEASURE_KER_TIME(src_.getUMat().convertTo(res, workType));

    if (singleFilterCall || numIters == 1)
    {
        adistHor = a0distHor;
        adistVert = a0distVert;
    }
    else
    {
        a0distHor.copyTo(adistHor);
        a0distVert.copyTo(adistVert);
    }

    if (true)
    {
        filterRF_blockAlg(res, resT, adistHor, adistVert);
    }
    else
    {
        filterRF_naiveAlg(res, resT, adistHor, adistVert);
    }

    res.convertTo(dst_, dDepth);
}

void DTFilterOCL::filterRF_blockAlg(UMat& res, UMat& resT, UMat& adistHor, UMat& adistVert)
{
    resT.create(res.cols, res.rows, res.type());
    UMat weights = UMatAligned(res.rows, res.cols, CV_32FC1, getCacheLineSize(), USAGE_ALLOCATE_DEVICE_MEMORY);
    UMat weightsT = UMatAligned(res.cols, res.rows, CV_32FC1, getCacheLineSize(), USAGE_ALLOCATE_DEVICE_MEMORY);

    for (int iter = 1; iter <= numIters; iter++)
    {
        transpose(res, resT);
        filterRF_iter_vert_pass(resT, adistHor, weightsT);

        transpose(resT, res);
        filterRF_iter_vert_pass(res, adistVert, weights);

        if (iter < numIters)
        {
            cv::pow(adistHor, 2, adistHor);
            cv::pow(adistVert, 2, adistVert);
        }
    }
}

void DTFilterOCL::filterRF_iter_vert_pass(UMat& res, UMat& adist, UMat& weights)
{
    int blockSize = (int)std::sqrt((float)res.rows);
    int blocksCount = (res.rows + blockSize-1)/blockSize;
    bool sync = true;

    for (int passId = 0; passId <= 1; passId++)
    {
        {
            char *knames[] = {"filter_RF_block_init_fwd", "filter_RF_block_init_bwd"};
            ocl::Kernel kerInit(knames[passId], kerProgSrcFilter, buildOptionsFlt);
            CV_Assert(!kerInit.empty());
            kerInit.args(
                ocl::KernelArg::ReadWrite(res),
                ocl::KernelArg::ReadWriteNoSize(adist),
                ocl::KernelArg::WriteOnlyNoSize(weights),
                blockSize);

            size_t globSize[] = { blocksCount, res.cols };
            size_t localSize[] = { 1, getMaxWorkGropSize() };

            CV_Assert(kerInit.run(2, globSize, localSize, sync));
            //imwrite(String(knames[passId]) + ".png", res.t());
        }

        {
            const char *knames[] = {"filter_RF_block_fill_borders_fwd", "filter_RF_block_fill_borders_bwd"};
            ocl::Kernel kerFillBorders(knames[passId], kerProgSrcFilter, buildOptionsFlt);
            CV_Assert(!kerFillBorders.empty());
            kerFillBorders.args(
                ocl::KernelArg::ReadWrite(res),
                ocl::KernelArg::ReadOnlyNoSize(weights),
                blockSize);

            size_t globSize[] = { res.cols };
            CV_Assert(kerFillBorders.run(1, globSize, NULL, sync));
            //imwrite(String(knames[passId]) + ".png", res.t());
        }

        {
            const char *knames[] = {"filter_RF_block_fill_fwd", "filter_RF_block_fill_bwd"};
            ocl::Kernel kerFill(knames[passId], kerProgSrcFilter, buildOptionsFlt);
            CV_Assert(!kerFill.empty());
            kerFill.args(
                ocl::KernelArg::ReadWrite(res),
                ocl::KernelArg::ReadOnlyNoSize(weights),
                blockSize);

            size_t globSize[] = { res.rows - blockSize, res.cols };
            size_t localSize[] = { 1, getMaxWorkGropSize() };

            CV_Assert(kerFill.run(2, globSize, localSize, sync));
            //imwrite(String(knames[passId]) + ".png", res.t());
        }
    }
}

void DTFilterOCL::filterRF_naiveAlg(UMat& res, UMat& resT, UMat& adistHor, UMat& adistVert)
{
    ocl::Kernel kerHor("filter_RF_vert", kerProgSrcFilter, buildOptionsFlt);
    ocl::Kernel kerVert("filter_RF_vert", kerProgSrcFilter, buildOptionsFlt);
    CV_Assert(!kerHor.empty() && !kerVert.empty());

    size_t globSizeHor[] = { h };
    size_t globSizeVert[] = { w };
    bool syncQueue = true;

    for (int iter = 1; iter <= numIters; iter++)
    {
        bool statusHorPass, statusVertPass;
        int write_a0 = (iter < numIters);

        MEASURE_KER_TIME(transpose(res, resT));
        kerHor.args(ocl::KernelArg::ReadWrite(resT), ocl::KernelArg::ReadWriteNoSize(adistHor), write_a0);
        MEASURE_KER_TIME(statusHorPass = kerHor.run(1, globSizeHor, NULL, syncQueue));

        MEASURE_KER_TIME(transpose(resT, res));
        kerVert.args(ocl::KernelArg::ReadWrite(res), ocl::KernelArg::ReadWriteNoSize(adistVert), write_a0);
        MEASURE_KER_TIME(statusVertPass = kerVert.run(1, globSizeVert, NULL, syncQueue));

        CV_Assert(statusHorPass && statusVertPass);
    }
}

void DTFilterOCL::filterNC_IndexAlg(UMat& src, UMat& dst)
{
    CV_Assert(src.rows == h && src.cols == w);

    UMat srcT = UMatAligned(w, h, workType);
    UMat isrc, isrcT;
    
    ocl::Kernel kerFiltHor = ocl::Kernel("filter_NC_hor_by_bounds", kerProgSrcFilter, buildOptionsFlt);
    ocl::Kernel kerFiltVert = ocl::Kernel("filter_NC_hor_by_bounds", kerProgSrcFilter, buildOptionsFlt);

    size_t globSizeHor[] = {h, w};
    size_t globSizeVert[] = {w, h};
    size_t localSize[] = {1, getMaxWorkGropSize()};
    bool sync = true;
    
    MEASURE_KER_TIME( transpose(src, srcT) );
    for (int iter = 0; iter < numIters; iter++)
    {
        bool kerStatus = true;

        MEASURE_KER_TIME( integrateCols(srcT, isrcT) );
        MEASURE_KER_TIME( transpose(isrcT, isrc) );
        kerFiltHor.args(
            ocl::KernelArg::ReadOnlyNoSize(isrc),
            ocl::KernelArg::ReadOnlyNoSize(distIndexHor),
            ocl::KernelArg::WriteOnly(src),
            iter
            );
        MEASURE_KER_TIME( kerStatus &= kerFiltHor.run(2, globSizeHor, localSize, sync) );

        MEASURE_KER_TIME( integrateCols(src, isrc) );
        MEASURE_KER_TIME( transpose(isrc, isrcT) );
        kerFiltVert.args(
            ocl::KernelArg::ReadOnlyNoSize(isrcT),
            ocl::KernelArg::ReadOnlyNoSize(distIndexVert),
            ocl::KernelArg::WriteOnly(srcT),
            iter
            );
        MEASURE_KER_TIME( kerStatus &= kerFiltVert.run(2, globSizeVert, localSize, sync) );

        CV_Assert(kerStatus);
    }
    MEASURE_KER_TIME( transpose(srcT, dst) );
}

void DTFilterOCL::filterNC_perRowAlg(UMat& src, UMat& dst)
{
    dst.create(h, w, workType);
    UMat& res = dst;
    UMat srcT = UMat(w, h, workType);
    UMat resT = UMat(w, h, workType);
    MEASURE_KER_TIME(transpose(src, srcT));

    ocl::Kernel kerHor("filter_NC_vert", kerProgSrcFilter, buildOptionsFlt);
    ocl::Kernel kerVert("filter_NC_vert", kerProgSrcFilter, buildOptionsFlt);
    CV_Assert(!kerHor.empty() && !kerVert.empty());

    size_t globSizeHor[] = { h };
    size_t globSizeVert[] = { w };
    bool syncQueue = true;

    for (int iter = 1; iter <= numIters; iter++)
    {
        bool statusHorPass, statusVertPass;
        float radius = getIterRadius(iter);

        kerHor.args(
            ocl::KernelArg::ReadOnly(srcT),
            ocl::KernelArg::ReadOnlyNoSize(idistHor),
            ocl::KernelArg::WriteOnlyNoSize(resT),
            radius
            );
        MEASURE_KER_TIME(statusHorPass = kerHor.run(1, globSizeHor, NULL, syncQueue));
        MEASURE_KER_TIME(transpose(resT, src));

        kerVert.args(
            ocl::KernelArg::ReadOnly(src),
            ocl::KernelArg::ReadOnlyNoSize(idistVert),
            ocl::KernelArg::WriteOnlyNoSize(res),
            radius
            );
        MEASURE_KER_TIME(statusVertPass = kerVert.run(1, globSizeVert, NULL, syncQueue));
        if (iter < numIters)
            MEASURE_KER_TIME(transpose(res, srcT));

        CV_Assert(statusHorPass && statusVertPass);
    }
}

void DTFilterOCL::filterNC_PerPixelAlg(UMat& src, UMat& dst)
{
    dst = src;
    UMat res(h, w, workType);
    UMat srcT(w, h, workType), resT(w, h, workType);

    ocl::Kernel kerHor("filter_NC_hor", kerProgSrcFilter);
    ocl::Kernel kerVert("filter_NC_hor", kerProgSrcFilter);
    CV_Assert(!kerHor.empty() && !kerVert.empty());

    size_t globSizeHor[] = { h, w };
    size_t globSizeVert[] = { w, h };
    size_t localSize[] = { 1, getMaxWorkGropSize() };
    bool sync = true;

    for (int iter = 1; iter <= numIters; iter++)
    {
        bool kerStatus = true;
        float radius = getIterRadius(iter);

        kerHor.args(
            ocl::KernelArg::ReadOnly(src),
            ocl::KernelArg::ReadOnlyNoSize(idistHor),
            ocl::KernelArg::WriteOnlyNoSize(res),
            radius
            );
        MEASURE_KER_TIME(kerStatus &= kerHor.run(2, globSizeHor, localSize, sync));
        MEASURE_KER_TIME(transpose(res, srcT));

        kerVert.args(
            ocl::KernelArg::ReadOnly(srcT),
            ocl::KernelArg::ReadOnlyNoSize(idistVert),
            ocl::KernelArg::WriteOnlyNoSize(resT),
            radius
            );
        MEASURE_KER_TIME(kerStatus &= kerVert.run(2, globSizeVert, localSize, sync));
        MEASURE_KER_TIME(transpose(resT, src));

        CV_Assert(kerStatus);
    }
}

void DTFilterOCL::filterIC(InputArray src_, OutputArray dst_, int dDepth)
{
    UMat srcWrapper(h + 1, w + 2, workType);
    UMat srcTWrapper(w + 1, h + 2, workType);
    UMat src = srcWrapper(Range(1, 1 + h), Range(1, 1 + w));
    UMat srcT = srcTWrapper(Range(1, 1 + w), Range(1, 1 + h));
    UMat isrc = UMatAligned(h, w + 1, workType);
    UMat isrcT = UMatAligned(w + 1, h, workType);

    UMat res = UMatAligned(h, w, workType);
    UMat resT = UMatAligned(w, h, workType);
        
    MEASURE_KER_TIME( src_.getUMat().convertTo(src, workType) );

    ocl::Kernel kerHor = ocl::Kernel("filter_IC_hor", kerProgSrcFilter, buildOptionsFlt);
    ocl::Kernel kerVert = ocl::Kernel("filter_IC_hor", kerProgSrcFilter, buildOptionsFlt);
    CV_Assert(!kerHor.empty() && !kerVert.empty());
    
    size_t globSizeHor[] = {h, w};
    size_t globSizeVert[] = {w, h};
    size_t localSize[] = {1, getMaxWorkGropSize()};
    bool sync = true;

    MEASURE_KER_TIME( transpose(src, resT) );
    for (int iter = 0; iter < numIters; iter++)
    {
        bool oclKerStatus = true;
        float curIterRadius = getIterRadius(iter);

        integrateColsIC(resT, distHorT, isrcT);
        MEASURE_KER_TIME( transpose(isrcT, isrc); );
        if (iter != 0) MEASURE_KER_TIME( transpose(resT, src); );
        duplicateVerticalBorders(srcWrapper);

        kerHor.args(
            ocl::KernelArg::ReadOnly(src),
            ocl::KernelArg::ReadOnlyNoSize(isrc),
            ocl::KernelArg::ReadOnlyNoSize(distIndexHor),
            ocl::KernelArg::ReadOnlyNoSize(tailHor),
            ocl::KernelArg::ReadOnlyNoSize(distHor),
            ocl::KernelArg::WriteOnlyNoSize(res),
            curIterRadius, iter
            );
        MEASURE_KER_TIME( oclKerStatus &= kerHor.run(2, globSizeHor, localSize, sync); );

        integrateColsIC(res, distVert, isrc);
        MEASURE_KER_TIME( transpose(isrc, isrcT); );
        MEASURE_KER_TIME( transpose(res, srcT); );
        duplicateVerticalBorders(srcTWrapper);
        kerVert.args(
            ocl::KernelArg::ReadOnly(srcT),
            ocl::KernelArg::ReadOnlyNoSize(isrcT),
            ocl::KernelArg::ReadOnlyNoSize(distIndexVert),
            ocl::KernelArg::ReadOnlyNoSize(tailVert),
            ocl::KernelArg::ReadOnlyNoSize(distVertT),
            ocl::KernelArg::WriteOnlyNoSize(resT),
            curIterRadius, iter
            );
        MEASURE_KER_TIME( oclKerStatus &= kerVert.run(2, globSizeVert, localSize, sync); );

        CV_Assert(oclKerStatus);
    }

    if (dDepth == resT.depth())
    {
        transpose(resT, dst_);
    }
    else
    {
        transpose(resT, res);
        res.convertTo(dst_, dDepth);
    }
}

void DTFilterOCL::setBuildOptionsFlt(InputArray src_)
{
    int cn      = src_.channels();
    //int depth   = src_.depth();
    workType    = CV_MAKE_TYPE(CV_32F, cn);

    buildOptionsFlt = format(
        "-D cn=%d "
        "-D SrcVec=%s "
        "-D NUM_ITERS=%d "
        ,
        cn,
        ocl::typeToStr(workType),
        numIters
        );
    //std::cout << "build options flt.: " << buildOptionsFlt.c_str() << "\n";
}

void DTFilterOCL::setBuildOptionsDT(UMat &guide)
{
    int gType = guide.type();
    int gcn = guide.channels();
    int gDepth = guide.depth();

    char strBuf[40];
    buildOptionsDT = format(
        "-D NUM_ITERS=%d "
        "-D cn=%d "
        "-D GuideType=%s "
        "-D GuideVec=%s "
        ,
        numIters,
        gcn,
        ocl::typeToStr(gDepth),
        ocl::typeToStr(gType)
        );

    if (gDepth != CV_32F)
    {
        buildOptionsDT += format("-D convert_guide=%s ", ocl::convertTypeStr(gDepth, CV_32F, gcn, strBuf));
    }
    //cout << "buildOptions:" << buildOptionsDT.c_str() << "\n";
}

void DTFilterOCL::computeDT(UMat& guide, UMat& distHorOuter, UMat& distVertOuter)
{
    ocl::Kernel kerHor = ocl::Kernel("compute_dt_hor", kerProgSrcDT, buildOptionsDT);
    ocl::Kernel kerVert = ocl::Kernel("compute_dt_vert", kerProgSrcDT, buildOptionsDT);
    CV_Assert(!kerHor.empty() && !kerVert.empty());

    distHorOuter.create(h, w + 1, CV_32FC1);
    distVertOuter.create(h + 1, w, CV_32FC1);
    UMat distHor = distHorOuter(Range::all(), Range(1, w + 1));
    UMat distVert = distVertOuter(Range(1, h + 1), Range::all());
    float maxRadius = 1.1f*getIterRadius(1);
    float sigmaRatio = (float)(sigmaSpatial/sigmaColor);

    kerHor.args(
        ocl::KernelArg::ReadOnly(guide),
        ocl::KernelArg::WriteOnlyNoSize(distHor),
        sigmaRatio,
        maxRadius
        );
    kerVert.args(
        ocl::KernelArg::ReadOnly(guide),
        ocl::KernelArg::WriteOnlyNoSize(distVert),
        sigmaRatio,
        maxRadius
        );

    size_t globSizeHor[] = {h, w + 1};
    size_t globSizeVert[] = {h + 1, w};
    size_t localSize[] = {1, getMaxWorkGropSize()};
    bool sync = true;

    bool oclKernelStatus = true;
    MEASURE_KER_TIME( oclKernelStatus &= kerHor.run(2, globSizeHor, localSize, sync); );
    MEASURE_KER_TIME( oclKernelStatus &= kerVert.run(2, globSizeVert, localSize, sync); );
    CV_Assert(oclKernelStatus);
}

void DTFilterOCL::computeIDT(UMat& guide, UMat& guideT)
{
    ocl::Kernel kerHor("compute_idt_vert", kerProgSrcDT, buildOptionsDT);
    ocl::Kernel kerVert("compute_idt_vert", kerProgSrcDT, buildOptionsDT);
    CV_Assert(!kerHor.empty() && !kerVert.empty());

    UMat idistHorWrap(w + 2, h, CV_32FC1);
    UMat idistVertWrap(h + 2, w, CV_32FC1);
    idistHor = idistHorWrap(Range(1, 1 + w), Range::all());
    idistVert = idistVertWrap(Range(1, 1 + h), Range::all());
    float sigmaRatio = (float)(sigmaSpatial / sigmaColor);

    kerHor.args(ocl::KernelArg::ReadOnly(guideT),
        ocl::KernelArg::WriteOnlyNoSize(idistHor),
        sigmaRatio);

    kerVert.args(ocl::KernelArg::ReadOnly(guide),
        ocl::KernelArg::WriteOnlyNoSize(idistVert),
        sigmaRatio);

    size_t globalSizeHor[] = { guide.rows };
    size_t globalSizeVert[] = { guide.cols };
    bool sync = true;

    MEASURE_KER_TIME(kerHor.run(1, globalSizeHor, NULL, sync));
    MEASURE_KER_TIME(kerVert.run(1, globalSizeVert, NULL, sync));

    if (NC_ocl_implememtation != ALG_PER_ROW)
    {
        MEASURE_KER_TIME(transpose(idistHorWrap, idistHorWrap));
        MEASURE_KER_TIME(transpose(idistVertWrap, idistVertWrap));
        idistHor = idistHorWrap(Range::all(), Range(1, 1 + w));
        idistVert = idistVertWrap(Range::all(), Range(1, 1 + h));
    }
    else
    {
        idistHor = idistHorWrap(Range(1, 1 + w), Range::all());
        idistVert = idistVertWrap(Range(1, 1 + h), Range::all());
    }
}

void DTFilterOCL::duplicateVerticalBorders(UMat& srcOuter)
{
    Range rangeRows(0, srcOuter.rows);
    
    {
        UMat leftBorderSrc = srcOuter(rangeRows, Range(1, 2));
        UMat leftBorderDst = srcOuter(rangeRows, Range(0, 1));
        leftBorderSrc.copyTo(leftBorderDst);
    }

    {
        UMat rightBorderSrc = srcOuter(rangeRows, Range(srcOuter.cols - 2, srcOuter.cols - 1));
        UMat rightBorderDst = srcOuter(rangeRows, Range(srcOuter.cols - 1, srcOuter.cols));
        rightBorderSrc.copyTo(rightBorderDst);
    }
}

void DTFilterOCL::integrateColsIC(UMat& src, UMat& dist, UMat& isrc)
{
    CV_Assert(src.size() == dist.size() && src.depth() == CV_32F);
    isrc.create(src.rows + 1, src.cols, workType);

    ocl::Kernel ker("integrate_cols_with_dist", kerProgSrcFilter, buildOptionsFlt);
    CV_Assert(!ker.empty());

    ker.args(
        ocl::KernelArg::ReadOnly(src),
        ocl::KernelArg::ReadOnlyNoSize(dist),
        ocl::KernelArg::WriteOnlyNoSize(isrc)
        );

    size_t globSize[] = {src.cols};
    size_t localSize[] = {getMaxWorkGropSize()};
    bool oclKernelStatus;

    MEASURE_KER_TIME( oclKernelStatus = ker.run(1, globSize, localSize, true) );
    CV_Assert(oclKernelStatus);
}

cv::UMat DTFilterOCL::UMatAligned(int rows, int cols, int type, int align, int usageFlags)
{
    int depth       = CV_MAT_DEPTH(type);
    int cn          = CV_MAT_CN(type);
    int elemSize1   = CV_ELEM_SIZE1(type);
    int elemSize    = CV_ELEM_SIZE(type);
    int lineSize    = ((cols*elemSize + align-1) / align)*align;
    CV_Assert(lineSize % elemSize1 == 0);

    UMat dst(rows, lineSize/elemSize1, CV_MAKE_TYPE(depth, 1), usageFlags);
    return dst(Range::all(), Range(0, cols*cn)).reshape(cn);
}

void DTFilterOCL::createAligned(UMat& src, int rows, int cols, int type, int align, int usageFlags)
{
    if (src.empty() || src.rows != rows || src.cols != cols || src.type() != type)
    {
        if (CV_ELEM_SIZE(type)*cols % align == 0)
        {
            src.create(rows, cols, type, (UMatUsageFlags)usageFlags);
        }
        else
        {
            src = UMatAligned(rows, cols, type, align, usageFlags);
        }
    }
}

void DTFilterOCL::setSingleFilterCall(bool flag)
{
    this->singleFilterCall = flag;
}

}