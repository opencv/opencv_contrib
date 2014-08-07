#pragma once
#include "precomp.hpp"

namespace cv
{

class DTFilterOCL : public DTFilter
{
public:

    static Ptr<DTFilterOCL> create(InputArray guide, double sigmaSpatial, double sigmaColor, int mode = DTF_NC, int numIters = 3);

    void filter(InputArray src, OutputArray dst, int dDepth = -1);

    void setSingleFilterCall(bool flag);

    ~DTFilterOCL();  

protected: /*Members declarations*/

    UMat idistHor, idistVert;
    UMat distHor, distVert;
    UMat distHorT, distVertT;
    UMat &a0distHor, &a0distVert; //synonyms of distHor distVert

    UMat distIndexHor, distIndexVert;
    UMat tailVert, tailHor;

    int mode, numIters;
    float sigmaSpatial, sigmaColor;

    bool singleFilterCall;
    int h, w;
    int workType;

    double meanDist;

    static int64 totalKernelsTime;
    int64 kernelsTime;

    ocl::ProgramSource kerProgSrcDT;
    ocl::ProgramSource kerProgSrcFilter;

    cv::String buildOptionsFlt;
    cv::String buildOptionsDT;

    int NC_ocl_implememtation;
    enum NCOclImplementation
    {
        ALG_PER_ROW,
        ALG_PER_PIXEL,
        ALG_INDEX,
    };

protected: /*Functions declarations*/

    DTFilterOCL();

    void init(InputArray guide, double sigmaSpatial, double sigmaColor, int mode = DTF_NC, int numIters = 3);

    inline double getIterSigmaH(int iterNum)
    {
        return sigmaSpatial * std::pow(2.0, numIters - iterNum) / sqrt(std::pow(4.0, numIters) - 1);
    }

    inline float getIterRadius(int iterNum)
    {
        return (float)(3.0*getIterSigmaH(iterNum));
    }

    inline float getIterAlpha(int iterNum)
    {
        return (float)std::exp(-std::sqrt(2.0 / 3.0) / getIterSigmaH(iterNum));
    }

    int getMaxWorkGropSize();
    int getCacheLineSize();

    void initProgram();
    void setBuildOptionsFlt(InputArray src_);
    void setBuildOptionsDT(UMat& guide);

    void initDT_NC(UMat& guide);
    void initDT_RF(UMat& guide);
    void initDT_IC(UMat& guide);

    void computeDT(UMat& guide, UMat& distHorOuter, UMat& distVertOuter);
    void computeIDT(UMat& guide, UMat& guideT);
    
    void filterNC_IndexAlg(UMat& src, UMat& dst);
    void filterNC_perRowAlg(UMat& src, UMat& dst);
    void filterNC_PerPixelAlg(UMat& src, UMat& dst);
    void filterIC(InputArray src_, OutputArray dst_, int dDepth);
    void filterRF(InputArray src_, OutputArray dst_, int dDepth);

    void filterRF_naiveAlg(UMat& res, UMat& resT, UMat& adistHor, UMat& adistVert);

    void filterRF_blockAlg(UMat& res, UMat& resT, UMat& adistHor, UMat& adistVert);
    void filterRF_iter_vert_pass(UMat& res, UMat& adist, UMat& weights);

    void computeIndexMat(UMat& idist, UMat& distIndex, bool useIDT = true);
    void integrateCols(UMat& src, UMat& isrc);
    
    void computeIndexAndTailMat(UMat& dist, UMat& distIndex, UMat& tail);
    void duplicateVerticalBorders(UMat& srcOuter);
    void integrateColsIC(UMat& src, UMat& dist, UMat& isrc);

    static UMat UMatAligned(int rows, int cols, int type, int align = 16, int useageFlags = USAGE_DEFAULT);
    static void createAligned(UMat& src, int rows, int cols, int type, int align = 16, int usageFlags = USAGE_DEFAULT);
};

}