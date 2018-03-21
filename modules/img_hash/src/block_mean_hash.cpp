// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

using namespace cv;
using namespace cv::img_hash;
using namespace std;

namespace {

enum
{
    imgWidth = 256,
    imgHeight = 256,
    blockWidth = 16,
    blockHeigth = 16,
    blockPerCol = imgHeight / blockHeigth,
    blockPerRow = imgWidth / blockWidth,
    rowSize = imgHeight - blockHeigth,
    colSize = imgWidth - blockWidth
};

class BlockMeanHashImpl CV_FINAL : public ImgHashBase::ImgHashImpl
{
public:
    BlockMeanHashImpl(int mode)
    {
        setMode(mode);
    }

    ~BlockMeanHashImpl() CV_OVERRIDE {}

    virtual void compute(cv::InputArray inputArr, cv::OutputArray outputArr) CV_OVERRIDE
    {
        cv::Mat const input = inputArr.getMat();
        CV_Assert(input.type() == CV_8UC4 ||
                  input.type() == CV_8UC3 ||
                  input.type() == CV_8U);

        cv::resize(input, resizeImg_, cv::Size(imgWidth,imgHeight), 0, 0, INTER_LINEAR_EXACT);
        if(input.type() == CV_8UC3)
        {
            cv::cvtColor(resizeImg_, grayImg_, CV_BGR2GRAY);
        }
        else if(input.type() == CV_8UC4)
        {
            cv::cvtColor(resizeImg_, grayImg_, CV_BGRA2GRAY);
        }
        else
        {
            grayImg_ = resizeImg_;
        }

        int pixColStep = blockWidth;
        int pixRowStep = blockHeigth;
        int numOfBlocks = 0;
        switch(mode_)
        {
        case BLOCK_MEAN_HASH_MODE_0:
        {
            numOfBlocks = blockPerCol * blockPerRow;
            break;
        }
        case BLOCK_MEAN_HASH_MODE_1:
        {
            pixColStep /= 2;
            pixRowStep /= 2;
            numOfBlocks = (blockPerCol*2-1) * (blockPerRow*2-1);
            break;
        }
        default:
            break;
        }

        mean_.resize(numOfBlocks);
        findMean(pixRowStep, pixColStep);
        outputArr.create(1, numOfBlocks/8 + numOfBlocks % 8, CV_8U);
        cv::Mat hash = outputArr.getMat();
        createHash(hash);
    }

    virtual double compare(cv::InputArray hashOne, cv::InputArray hashTwo) const CV_OVERRIDE
    {
        return norm(hashOne, hashTwo, NORM_HAMMING);
    }

    void setMode(int mode)
    {
        CV_Assert(mode == BLOCK_MEAN_HASH_MODE_0 || mode == BLOCK_MEAN_HASH_MODE_1);
        mode_ = mode;
    }

    void createHash(cv::Mat &hash)
    {
        double const median = cv::mean(grayImg_)[0];
        uchar *hashPtr = hash.ptr<uchar>(0);
        std::bitset<8> bits = 0;
        for(size_t i = 0; i < mean_.size(); ++i)
        {
            size_t const residual = i%8;
            bits[residual] = mean_[i] < median ? 0 : 1;
            if(residual == 7)
            {
                *hashPtr = static_cast<uchar>(bits.to_ulong());
                ++hashPtr;
            }else if(i == mean_.size() - 1)
            {
                *hashPtr = bits[residual];
            }
        }
    }
    void findMean(int pixRowStep, int pixColStep)
    {
        size_t blockIdx = 0;
        for(int row = 0; row <= rowSize; row += pixRowStep)
        {
            for(int col = 0; col <= colSize; col += pixColStep)
            {
                mean_[blockIdx++] = cv::mean(grayImg_(cv::Rect(col, row, blockWidth, blockHeigth)))[0];
            }
        }
    }

    cv::Mat grayImg_;
    std::vector<double> mean_;
    int mode_;
    cv::Mat resizeImg_;
};

inline BlockMeanHashImpl *getLocalImpl(ImgHashBase::ImgHashImpl *ptr)
{
    BlockMeanHashImpl * impl = static_cast<BlockMeanHashImpl*>(ptr);
    CV_Assert(impl);
    return impl;
}

}

//==================================================================================================

namespace cv { namespace img_hash {

Ptr<BlockMeanHash> BlockMeanHash::create(int mode)
{
    Ptr<BlockMeanHash> res(new BlockMeanHash);
    res->pImpl = makePtr<BlockMeanHashImpl>(mode);
    return res;
}

void BlockMeanHash::setMode(int mode)
{
    getLocalImpl(pImpl)->setMode(mode);
}

std::vector<double> BlockMeanHash::getMean() const
{
    return getLocalImpl(pImpl)->mean_;
}

void blockMeanHash(cv::InputArray inputArr, cv::OutputArray outputArr, int mode)
{
    BlockMeanHashImpl(mode).compute(inputArr, outputArr);
}

}} // cv::img_hash::
