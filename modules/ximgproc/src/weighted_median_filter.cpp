#include "precomp.hpp"
#include <vector>
using std::vector;

namespace cv
{

class WeightedMedianImpl : public WeightedMedian
{
public:

    void filter(InputArray src, OutputArray dst, double rangeQuantizer, int dDepth);

    WeightedMedianImpl(InputArray guide, double spatialSize, double colorSize, int convFilter);

protected:

    Ptr<DTFilter> dtf;
    Ptr<GuidedFilter> gf;
    
    Size sz;
    bool isDT;

    class UpdateMedian_ParBody : public ParallelLoopBody
    {
        Mat &sum, &reached, &res;
        float medianLevel;
        int curLevelValue;

    public:
        UpdateMedian_ParBody(Mat& sum_, Mat& reached_, Mat& res_, int curLevelValue_, float medianLevel_ = 127.5f)
            : sum(sum_), reached(reached_), res(res_), curLevelValue(curLevelValue_), medianLevel(medianLevel_) {}

        void operator() (const Range& range) const;
    };

    class ComputeMedian_ParBody : public ParallelLoopBody
    {
        vector<Mat>& layersRes;
        Mat &sum, &res;
        int layersStart, layerSize;

    public:
        ComputeMedian_ParBody(vector<Mat>& layersRes_, Mat& sum_, Mat& res_, int layersStart_, int layerSize_)
            :layersRes(layersRes_), sum(sum_), res(res_), layersStart(layersStart_), layerSize(layerSize_) {}

        void operator() (const Range& range) const;
    };
};

WeightedMedianImpl::WeightedMedianImpl(InputArray guide, double spatialSize, double colorSize, int filterType)
{
    isDT = (filterType == DTF_NC || filterType == DTF_IC || filterType == DTF_RF);

    if (isDT)
    {
        dtf = createDTFilter(guide, spatialSize, colorSize, filterType, 3);
    }
    else if (filterType == GUIDED_FILTER)
    {
        gf = createGuidedFilter(guide, cvRound(spatialSize), colorSize);
    }
    else
    {
        CV_Error(Error::StsBadFlag, "Unsupported type of edge aware filter (only guided filter and domain transform allowed)");
    }

    sz = guide.size();
}

void WeightedMedianImpl::filter(InputArray src, OutputArray dst, double rangeQuantizer, int dDepth)
{
    CV_Assert(src.size() == sz && src.depth() == CV_8U);
    
    int layerSize = cvRound(rangeQuantizer);
    int srcCnNum = src.channels();

    vector<Mat> srcCn(srcCnNum);
    vector<Mat> resCn(srcCnNum);
    if (srcCnNum == 1)
        srcCn[0] = src.getMat();
    else
        split(src, srcCn);

    Mat reached, sum;
    vector<Mat> resLayers;

    for (int cnId = 0; cnId < srcCnNum; cnId++)
    {
        double minVal, maxVal;
        minMaxLoc(srcCn[cnId], &minVal, &maxVal);
        
        int layerStart  = (int)minVal;
        int layerEnd    = cvRound(maxVal);
        int layersCount = (layerEnd - layerStart) / layerSize;
        
        sum = Mat::zeros(sz, CV_32FC1);

        if (isDT)
        {
            resCn[cnId] = Mat::zeros(sz, CV_8UC1);
            reached = Mat::zeros(sz, CV_8UC1);
        }
        else
        {
            resLayers.resize(layersCount);
        }
     
        Mat layer, layerFiltered;

        for (int layerId = 0; layerId < layersCount; layerId++)
        {
            int curLevelVal = layerStart + layerId*layerSize;
            
            if (src.depth() == CV_8U && layerSize == 1)
            {
                layer = (srcCn[cnId] == layerStart + layerId);
            }
            else
            {
                layer = (srcCn[cnId] >= curLevelVal) & (srcCn[cnId] < curLevelVal + layerSize);
            }

            if (isDT)
            {
                dtf->filter(layer, layerFiltered, CV_32F);
                add(layerFiltered, sum, sum);

                UpdateMedian_ParBody updateMedian(sum, reached, resCn[cnId], curLevelVal);
                parallel_for_(Range(0, sz.height), updateMedian);
            }
            else
            {
                gf->filter(layer, resLayers[layerId], CV_32F);
                add(sum, resLayers[layerId], sum);
            }
        }
        
        if (!isDT)
        {
            resCn[cnId].create(sz, CV_8UC1);
            
            ComputeMedian_ParBody computeMedian(resLayers, sum, resCn[cnId], layerStart, layerSize);
            parallel_for_(Range(0, sz.height), computeMedian);
        }
    }

    if (dDepth == -1) dDepth = src.depth();
    if (dDepth == CV_8U)
    {
        merge(resCn, dst);
    }
    else
    {
        Mat res;
        merge(resCn, res);
        res.convertTo(dst, dDepth);
    }
}

void WeightedMedianImpl::UpdateMedian_ParBody::operator()(const Range& range) const
{
    for (int i = range.start; i < range.end; i++)
    {
        float *sumLine = sum.ptr<float>(i);
        uchar *resLine = res.ptr<uchar>(i);
        uchar *reachedLine = reached.ptr<uchar>(i);

        for (int j = 0; j < sum.cols; j++)
        {
            if (reachedLine[j])
            {
                continue;
            }
            else if (sumLine[j] >= medianLevel)
            {
                resLine[j] = (uchar) curLevelValue;
                reachedLine[j] = 0xFF;
            }
        }
    }
}

void WeightedMedianImpl::ComputeMedian_ParBody::operator()(const Range& range) const
{
    for (int i = range.start; i < range.end; i++)
    {
        float *sumLine = sum.ptr<float>(i);
        uchar *resLine = res.ptr<uchar>(i);

        for (int j = 0; j < sum.cols; j++)
        {
            float curSum = 0.0f;
            float medianSum = sumLine[j] / 2.0f;

            int l = 0;
            for (l = 0; l < (int)layersRes.size(); l++)
            {
                if (curSum >= medianSum)
                    break;

                curSum += layersRes[l].at<float>(i, j);
            }

            resLine[j] = (uchar) (layersStart + l*layerSize);
        }
    }
}

CV_EXPORTS_W
Ptr<WeightedMedian> createWeightedMedianFilter(InputArray guide, double spatialSize, double colorSize, int filterType)
{
    return Ptr<WeightedMedian>(new WeightedMedianImpl(guide, spatialSize, colorSize, filterType));
}

CV_EXPORTS_W void weightedMedianFilter(InputArray guide, InputArray src, OutputArray dst, double spatialSize, double colorSize, int filterType, double rangeQuantizer)
{
    WeightedMedian *wm = new WeightedMedianImpl(guide, spatialSize, colorSize, filterType);
    wm->filter(src, dst, rangeQuantizer);
    delete wm;
}


}