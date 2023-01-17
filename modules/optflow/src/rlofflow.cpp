// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "precomp.hpp"
#include "rlof/rlof_localflow.h"
#include "rlof/geo_interpolation.hpp"
#include "opencv2/ximgproc.hpp"


namespace cv {
namespace optflow {

Ptr<RLOFOpticalFlowParameter> RLOFOpticalFlowParameter::create()
{
    return Ptr<RLOFOpticalFlowParameter>(new RLOFOpticalFlowParameter);
}

void RLOFOpticalFlowParameter::setUseMEstimator(bool val)
{
    if (val)
    {
        normSigma0 = 3.2f;
        normSigma1 = 7.f;
    }
    else
    {
        normSigma0 = std::numeric_limits<float>::max();
        normSigma1 = std::numeric_limits<float>::max();
    }
}
void RLOFOpticalFlowParameter::setSolverType(SolverType val){ solverType = val;}
SolverType RLOFOpticalFlowParameter::getSolverType() const { return solverType;}

void RLOFOpticalFlowParameter::setSupportRegionType(SupportRegionType val){ supportRegionType = val;}
SupportRegionType RLOFOpticalFlowParameter::getSupportRegionType() const { return supportRegionType;}

void RLOFOpticalFlowParameter::setNormSigma0(float val){ normSigma0 = val;}
float RLOFOpticalFlowParameter::getNormSigma0() const { return normSigma0;}

void RLOFOpticalFlowParameter::setNormSigma1(float val){ normSigma1 = val;}
float RLOFOpticalFlowParameter::getNormSigma1() const { return normSigma1;}

void RLOFOpticalFlowParameter::setSmallWinSize(int val){ smallWinSize = val;}
int RLOFOpticalFlowParameter::getSmallWinSize() const { return smallWinSize;}

void RLOFOpticalFlowParameter::setLargeWinSize(int val){ largeWinSize = val;}
int RLOFOpticalFlowParameter::getLargeWinSize() const { return largeWinSize;}

void RLOFOpticalFlowParameter::setCrossSegmentationThreshold(int val){ crossSegmentationThreshold = val;}
int RLOFOpticalFlowParameter::getCrossSegmentationThreshold() const { return crossSegmentationThreshold;}

void RLOFOpticalFlowParameter::setMaxLevel(int val){ maxLevel = val;}
int RLOFOpticalFlowParameter::getMaxLevel() const { return maxLevel;}

void RLOFOpticalFlowParameter::setUseInitialFlow(bool val){ useInitialFlow = val;}
bool RLOFOpticalFlowParameter::getUseInitialFlow() const { return useInitialFlow;}

void RLOFOpticalFlowParameter::setUseIlluminationModel(bool val){ useIlluminationModel = val;}
bool RLOFOpticalFlowParameter::getUseIlluminationModel() const { return useIlluminationModel;}

void RLOFOpticalFlowParameter::setUseGlobalMotionPrior(bool val){ useGlobalMotionPrior = val;}
bool RLOFOpticalFlowParameter::getUseGlobalMotionPrior() const { return useGlobalMotionPrior;}

void RLOFOpticalFlowParameter::setMaxIteration(int val){ maxIteration = val;}
int RLOFOpticalFlowParameter::getMaxIteration() const { return maxIteration;}

void RLOFOpticalFlowParameter::setMinEigenValue(float val){ minEigenValue = val;}
float RLOFOpticalFlowParameter::getMinEigenValue() const { return minEigenValue;}

void RLOFOpticalFlowParameter::setGlobalMotionRansacThreshold(float val){ globalMotionRansacThreshold = val;}
float RLOFOpticalFlowParameter::getGlobalMotionRansacThreshold() const { return globalMotionRansacThreshold;}

class DenseOpticalFlowRLOFImpl : public DenseRLOFOpticalFlow
{
public:
    DenseOpticalFlowRLOFImpl()
        : param(Ptr<RLOFOpticalFlowParameter>(new RLOFOpticalFlowParameter))
        , forwardBackwardThreshold(1.f)
        , gridStep(6, 6)
        , interp_type(InterpolationType::INTERP_GEO)
        , k(128)
        , sigma(0.05f)
        , lambda(999.f)
        , fgs_lambda(500.0f)
        , fgs_sigma(1.5f)
        , use_post_proc(true)
        , use_variational_refinement(false)
        , sp_size(15)
        , slic_type(ximgproc::SLIC)

    {
        prevPyramid[0] = cv::Ptr<CImageBuffer>(new CImageBuffer);
        prevPyramid[1] = cv::Ptr<CImageBuffer>(new CImageBuffer);
        currPyramid[0] = cv::Ptr<CImageBuffer>(new CImageBuffer);
        currPyramid[1] = cv::Ptr<CImageBuffer>(new CImageBuffer);
    }
    virtual void setRLOFOpticalFlowParameter(Ptr<RLOFOpticalFlowParameter>  val) CV_OVERRIDE { param = val; }
    virtual Ptr<RLOFOpticalFlowParameter>  getRLOFOpticalFlowParameter() const CV_OVERRIDE { return param; }

    virtual float getForwardBackward() const CV_OVERRIDE { return forwardBackwardThreshold; }
    virtual void setForwardBackward(float val) CV_OVERRIDE { forwardBackwardThreshold = val; }

    virtual void setInterpolation(InterpolationType val) CV_OVERRIDE { interp_type = val; }
    virtual InterpolationType getInterpolation() const CV_OVERRIDE { return interp_type; }

    virtual Size getGridStep() const CV_OVERRIDE { return gridStep; }
    virtual void setGridStep(Size val) CV_OVERRIDE { gridStep = val; }

    virtual int getEPICK() const CV_OVERRIDE { return k; }
    virtual void setEPICK(int val) CV_OVERRIDE { k = val; }

    virtual float getEPICSigma() const CV_OVERRIDE { return sigma; }
    virtual void setEPICSigma(float val) CV_OVERRIDE { sigma = val; }

    virtual float getEPICLambda() const CV_OVERRIDE { return lambda; }
    virtual void setEPICLambda(float val)  CV_OVERRIDE { lambda = val; }

    virtual float getFgsLambda() const CV_OVERRIDE { return fgs_lambda; }
    virtual void setFgsLambda(float val) CV_OVERRIDE { fgs_lambda = val; }

    virtual float getFgsSigma() const CV_OVERRIDE { return fgs_sigma; }
    virtual void setFgsSigma(float val) CV_OVERRIDE { fgs_sigma = val; }

    virtual bool getUsePostProc() const CV_OVERRIDE { return use_post_proc; }
    virtual void setUsePostProc(bool val) CV_OVERRIDE { use_post_proc = val; }

    virtual void setUseVariationalRefinement(bool val) CV_OVERRIDE { use_variational_refinement = val; }
    virtual bool getUseVariationalRefinement() const CV_OVERRIDE { return use_variational_refinement; }

    virtual void setRICSPSize(int val) CV_OVERRIDE { sp_size = val; }
    virtual int  getRICSPSize() const CV_OVERRIDE { return sp_size; }

    virtual void setRICSLICType(int val) CV_OVERRIDE { slic_type = static_cast<ximgproc::SLICType>(val); }
    virtual int  getRICSLICType() const CV_OVERRIDE { return slic_type; }

    virtual void calc(InputArray I0, InputArray I1, InputOutputArray flow) CV_OVERRIDE
    {
        CV_Assert(!I0.empty() && I0.depth() == CV_8U && (I0.channels() == 3 || I0.channels() == 1));
        CV_Assert(!I1.empty() && I1.depth() == CV_8U && (I1.channels() == 3 || I1.channels() == 1));
        CV_Assert(I0.sameSize(I1));
        if (param.empty())
            param = Ptr<RLOFOpticalFlowParameter>(new RLOFOpticalFlowParameter());
        if (param->supportRegionType == SR_CROSS)
            CV_Assert( I0.channels() == 3 && I1.channels() == 3);
        CV_Assert(interp_type == InterpolationType::INTERP_EPIC || interp_type == InterpolationType::INTERP_GEO || interp_type == InterpolationType::INTERP_RIC);
        // if no parameter is used use the default parameter

        Mat prevImage = I0.getMat();
        Mat currImage = I1.getMat();
        int noPoints = prevImage.cols * prevImage.rows;
        std::vector<cv::Point2f> prevPoints(noPoints);
        std::vector<cv::Point2f> currPoints, refPoints;
        noPoints = 0;
        cv::Size grid_h = gridStep / 2;
        for (int r = grid_h.height; r < prevImage.rows - grid_h.height; r += gridStep.height)
        {
            for (int c = grid_h.width; c < prevImage.cols - grid_h.width; c += gridStep.width)
            {
                prevPoints[noPoints++] = cv::Point2f(static_cast<float>(c), static_cast<float>(r));
            }
        }
        prevPoints.erase(prevPoints.begin() + noPoints, prevPoints.end());
        currPoints.resize(prevPoints.size());
        calcLocalOpticalFlow(prevImage, currImage, prevPyramid, currPyramid, prevPoints, currPoints, *(param.get()));
        flow.create(prevImage.size(), CV_32FC2);
        Mat dense_flow = flow.getMat();

        std::vector<Point2f> filtered_prevPoints;
        std::vector<Point2f> filtered_currPoints;
        if (gridStep == cv::Size(1, 1) && forwardBackwardThreshold <= 0)
        {
            for (unsigned int n = 0; n < prevPoints.size(); n++)
            {
                dense_flow.at<Point2f>(prevPoints[n]) = currPoints[n] - prevPoints[n];
            }
            return;
        }
        if (forwardBackwardThreshold > 0)
        {
            // reuse image pyramids
            calcLocalOpticalFlow(currImage, prevImage, currPyramid, prevPyramid, currPoints, refPoints, *(param.get()));

            filtered_prevPoints.resize(prevPoints.size());
            filtered_currPoints.resize(prevPoints.size());
            float sqrForwardBackwardThreshold = forwardBackwardThreshold * forwardBackwardThreshold;
            noPoints = 0;
            for (unsigned int r = 0; r < refPoints.size(); r++)
            {
                Point2f diff = refPoints[r] - prevPoints[r];
                if (diff.x * diff.x + diff.y * diff.y < sqrForwardBackwardThreshold)
                {
                    filtered_prevPoints[noPoints] = prevPoints[r];
                    filtered_currPoints[noPoints++] = currPoints[r];
                }
            }

            filtered_prevPoints.erase(filtered_prevPoints.begin() + noPoints, filtered_prevPoints.end());
            filtered_currPoints.erase(filtered_currPoints.begin() + noPoints, filtered_currPoints.end());

        }
        else
        {
            filtered_prevPoints = prevPoints;
            filtered_currPoints = currPoints;
        }
        // Interpolators below expect non empty matches
        if (filtered_prevPoints.empty()) {
            flow.setTo(0);
            return;
        }
        if (interp_type == InterpolationType::INTERP_EPIC)
        {
            Ptr<ximgproc::EdgeAwareInterpolator> gd = ximgproc::createEdgeAwareInterpolator();
            gd->setK(k);
            gd->setSigma(sigma);
            gd->setLambda(lambda);
            gd->setFGSLambda(fgs_lambda);
            gd->setFGSSigma(fgs_sigma);
            gd->setUsePostProcessing(use_post_proc);
            gd->interpolate(prevImage, filtered_prevPoints, currImage, filtered_currPoints, dense_flow);
        }
        else if (interp_type == InterpolationType::INTERP_RIC)
        {
            Ptr<ximgproc::RICInterpolator> gd = ximgproc::createRICInterpolator();
            gd->setK(k);
            gd->setFGSLambda(fgs_lambda);
            gd->setFGSSigma(fgs_sigma);
            gd->setSuperpixelSize(sp_size);
            gd->setSuperpixelMode(slic_type);
            gd->setUseGlobalSmootherFilter(use_post_proc);
            gd->setUseVariationalRefinement(false);
            gd->interpolate(prevImage, filtered_prevPoints, currImage, filtered_currPoints, dense_flow);
        }
        else
        {
            Mat blurredPrevImage, blurredCurrImage;
            GaussianBlur(prevImage, blurredPrevImage, cv::Size(5, 5), -1);
            std::vector<uchar> status(filtered_currPoints.size(), 1);
            interpolate_irregular_nn_raster(filtered_prevPoints, filtered_currPoints, status, blurredPrevImage).copyTo(dense_flow);
            std::vector<Mat> vecMats;
            std::vector<Mat> vecMats2(2);
            cv::split(dense_flow, vecMats);
            cv::bilateralFilter(vecMats[0], vecMats2[0], 5, 2, 20);
            cv::bilateralFilter(vecMats[1], vecMats2[1], 5, 2, 20);
            cv::merge(vecMats2, dense_flow);
            if (use_post_proc)
            {
                ximgproc::fastGlobalSmootherFilter(prevImage, flow, flow, fgs_lambda, fgs_sigma);
            }
        }
        if (use_variational_refinement)
        {
            Mat prevGrey, currGrey;
            Ptr<VariationalRefinement > variationalrefine = VariationalRefinement::create();
            cvtColor(prevImage, prevGrey, COLOR_BGR2GRAY);
            cvtColor(currImage, currGrey, COLOR_BGR2GRAY);
            variationalrefine->setOmega(1.9f);
            variationalrefine->calc(prevGrey, currGrey, flow);
        }
    }

    virtual void collectGarbage() CV_OVERRIDE
    {
        prevPyramid[0].release();
        prevPyramid[1].release();
        currPyramid[0].release();
        currPyramid[1].release();
    }

protected:
    Ptr<RLOFOpticalFlowParameter> param;
    float                         forwardBackwardThreshold;
    Ptr<CImageBuffer>             prevPyramid[2];
    Ptr<CImageBuffer>             currPyramid[2];
    cv::Size                      gridStep;
    InterpolationType             interp_type;
    int                           k;
    float                         sigma;
    float                         lambda;
    float                         fgs_lambda;
    float                         fgs_sigma;
    bool                          use_post_proc;
    bool                          use_variational_refinement;
    int                           sp_size;
    ximgproc::SLICType            slic_type;
};

Ptr<DenseRLOFOpticalFlow> DenseRLOFOpticalFlow::create(
    Ptr<RLOFOpticalFlowParameter>  rlofParam,
    float forwardBackwardThreshold,
    cv::Size gridStep,
    InterpolationType interp_type,
    int epicK,
    float epicSigma,
    float epicLambda,
    int ricSPSize,
    int ricSLICType,
    bool use_post_proc,
    float fgs_lambda,
    float fgs_sigma,
    bool use_variational_refinement)
{
    Ptr<DenseRLOFOpticalFlow> algo = makePtr<DenseOpticalFlowRLOFImpl>();
    algo->setRLOFOpticalFlowParameter(rlofParam);
    algo->setForwardBackward(forwardBackwardThreshold);
    algo->setGridStep(gridStep);
    algo->setInterpolation(interp_type);
    algo->setEPICK(epicK);
    algo->setEPICSigma(epicSigma);
    algo->setEPICLambda(epicLambda);
    algo->setUsePostProc(use_post_proc);
    algo->setFgsLambda(fgs_lambda);
    algo->setFgsSigma(fgs_sigma);
    algo->setRICSLICType(ricSLICType);
    algo->setRICSPSize(ricSPSize);
    algo->setUseVariationalRefinement(use_variational_refinement);
    return algo;
}

class SparseRLOFOpticalFlowImpl : public SparseRLOFOpticalFlow
{
    public:
    SparseRLOFOpticalFlowImpl()
        : param(Ptr<RLOFOpticalFlowParameter>(new RLOFOpticalFlowParameter))
        , forwardBackwardThreshold(1.f)
    {
        prevPyramid[0] = cv::Ptr< CImageBuffer>(new CImageBuffer);
        prevPyramid[1] = cv::Ptr< CImageBuffer>(new CImageBuffer);
        currPyramid[0] = cv::Ptr< CImageBuffer>(new CImageBuffer);
        currPyramid[1] = cv::Ptr< CImageBuffer>(new CImageBuffer);
    }
    virtual void setRLOFOpticalFlowParameter(Ptr<RLOFOpticalFlowParameter>  val) CV_OVERRIDE { param = val; }
    virtual Ptr<RLOFOpticalFlowParameter>  getRLOFOpticalFlowParameter() const CV_OVERRIDE { return param; }

    virtual float getForwardBackward()  const CV_OVERRIDE { return forwardBackwardThreshold; }
    virtual void setForwardBackward(float val) CV_OVERRIDE { forwardBackwardThreshold = val; }

    virtual void calc(InputArray prevImg, InputArray nextImg,
        InputArray prevPts, InputOutputArray nextPts,
        OutputArray status,
        OutputArray err) CV_OVERRIDE
    {
        CV_Assert(!prevImg.empty() && prevImg.depth() == CV_8U && (prevImg.channels() == 3 || prevImg.channels() == 1));
        CV_Assert(!nextImg.empty() && nextImg.depth() == CV_8U && (nextImg.channels() == 3 || nextImg.channels() == 1));
        CV_Assert(prevImg.sameSize(nextImg));

        if (param.empty())
        {
            param = makePtr<RLOFOpticalFlowParameter>();
        }
        CV_DbgAssert(!param.empty());

        if (param->supportRegionType == SR_CROSS)
        {
            CV_CheckChannelsEQ(prevImg.channels(), 3, "SR_CROSS mode requires images with 3 channels");
            CV_CheckChannelsEQ(nextImg.channels(), 3, "SR_CROSS mode requires images with 3 channels");
        }

        Mat prevImage = prevImg.getMat();
        Mat nextImage = nextImg.getMat();
        Mat prevPtsMat = prevPts.getMat();

        if (param->useInitialFlow == false)
            nextPts.create(prevPtsMat.size(), prevPtsMat.type(), -1, true);

        int npoints = 0;
        CV_Assert((npoints = prevPtsMat.checkVector(2, CV_32F, true)) >= 0);
        if (npoints == 0)
        {
            nextPts.release();
            status.release();
            err.release();
            return;
        }
        Mat nextPtsMat = nextPts.getMat();
        CV_Assert(nextPtsMat.checkVector(2, CV_32F, true) == npoints);

        std::vector<cv::Point2f> prevPoints(npoints), nextPoints(npoints), refPoints;

        if (prevPtsMat.channels() != 2)
            prevPtsMat = prevPtsMat.reshape(2, npoints);

        prevPtsMat.copyTo(prevPoints);

        if (param->useInitialFlow )
        {
            if (nextPtsMat.channels() != 2)
                nextPtsMat = nextPtsMat.reshape(2, npoints);
            nextPtsMat.copyTo(nextPoints);
        }
        cv::Mat statusMat;
        cv::Mat errorMat;
        if (status.needed() || forwardBackwardThreshold > 0)
        {
            status.create((int)npoints, 1, CV_8U, -1, true);
            statusMat = status.getMat();
            statusMat.setTo(1);
        }

        if (err.needed() || forwardBackwardThreshold > 0)
        {
            err.create((int)npoints, 1, CV_32F, -1, true);
            errorMat = err.getMat();
            errorMat.setTo(0);
        }

        calcLocalOpticalFlow(prevImage, nextImage, prevPyramid, currPyramid, prevPoints, nextPoints, *(param.get()));
        cv::Mat(1,npoints , CV_32FC2, &nextPoints[0]).copyTo(nextPtsMat);
        if (forwardBackwardThreshold > 0)
        {
            // use temp variable to properly initialize refPoints
            // inside 'calcLocalOpticalFlow' when 'use_init_flow' and 'fwd_bwd_thresh' parameters are used
            bool temp_param = param->getUseInitialFlow();
            param->setUseInitialFlow(false);
            // reuse image pyramids
            calcLocalOpticalFlow(nextImage, prevImage, currPyramid, prevPyramid, nextPoints, refPoints, *(param.get()));
            param->setUseInitialFlow(temp_param);
        }
        for (unsigned int r = 0; r < refPoints.size(); r++)
        {
            Point2f diff = refPoints[r] - prevPoints[r];
            errorMat.at<float>(r) = sqrt(diff.x * diff.x + diff.y * diff.y);
            if (errorMat.at<float>(r) > forwardBackwardThreshold)
                statusMat.at<uchar>(r) = 0;
        }

    }

protected:
    Ptr<RLOFOpticalFlowParameter> param;
    float                forwardBackwardThreshold;
    Ptr<CImageBuffer>    prevPyramid[2];
    Ptr<CImageBuffer>    currPyramid[2];
};

Ptr<SparseRLOFOpticalFlow> SparseRLOFOpticalFlow::create(
    Ptr<RLOFOpticalFlowParameter>  rlofParam,
    float forwardBackwardThreshold)
{
    Ptr<SparseRLOFOpticalFlow> algo = makePtr<SparseRLOFOpticalFlowImpl>();
    algo->setRLOFOpticalFlowParameter(rlofParam);
    algo->setForwardBackward(forwardBackwardThreshold);
    return algo;
}

void calcOpticalFlowDenseRLOF(InputArray I0, InputArray I1, InputOutputArray flow,
    Ptr<RLOFOpticalFlowParameter>  rlofParam ,
    float forewardBackwardThreshold, Size gridStep,
    InterpolationType interp_type,
    int epicK, float epicSigma, float epicLambda,
    int superpixelSize, int superpixelType,
    bool use_post_proc, float fgsLambda, float fgsSigma, bool use_variational_refinement)
{
    Ptr<DenseRLOFOpticalFlow> algo = DenseRLOFOpticalFlow::create(
        rlofParam, forewardBackwardThreshold, gridStep, interp_type,
        epicK, epicSigma, epicLambda, superpixelSize, superpixelType,
        use_post_proc, fgsLambda, fgsSigma, use_variational_refinement);
    algo->calc(I0, I1, flow);
    algo->collectGarbage();
}

void calcOpticalFlowSparseRLOF(InputArray prevImg, InputArray nextImg,
    InputArray prevPts, InputOutputArray nextPts,
    OutputArray status, OutputArray err,
    Ptr<RLOFOpticalFlowParameter>  rlofParam,
    float forewardBackwardThreshold)
{
    Ptr<SparseRLOFOpticalFlow> algo = SparseRLOFOpticalFlow::create(
        rlofParam, forewardBackwardThreshold);
    algo->calc(prevImg, nextImg, prevPts, nextPts, status, err);
}
Ptr<DenseOpticalFlow> createOptFlow_DenseRLOF()
{
    return DenseRLOFOpticalFlow::create();
}

Ptr<SparseOpticalFlow> createOptFlow_SparseRLOF()
{
    return SparseRLOFOpticalFlow::create();
}

}} // namespace
