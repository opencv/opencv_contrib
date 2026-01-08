// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

using namespace cv;
using namespace cv::img_hash;
using namespace std;

namespace {

void getMHKernel(float alpha, float level, cv::Mat &kernel)
{
    int const sigma = static_cast<int>(4*std::pow(alpha,level));

    float const ratio = std::pow(alpha, -level);
    kernel.create(2*sigma+1, 2*sigma+1, CV_32F);
    for(int row = 0; row != kernel.rows; ++row)
    {
        float const ydiff = static_cast<float>(row - sigma);
        float const ypos = ratio * ydiff;
        float const yposPow2 = ypos * ypos;
        float *kPtr = kernel.ptr<float>(row);
        for(int col = 0; col != kernel.cols; ++col)
        {
            float const xpos = ratio * static_cast<float>((col - sigma));
            float const a = xpos * xpos + yposPow2;
            kPtr[col] = (2-a)*std::exp(a/2);
        }
    }
}

void fillBlocks(cv::Mat const &freImg, cv::Mat &blocks)
{
    //TODO : use forEach may provide better speed, however,
    //it is quite tedious to apply without lambda
    blocks.setTo(0);
    for(int row = 0; row != blocks.rows; ++row)
    {
        float *bptr = blocks.ptr<float>(row);
        int const rOffset = row*16;
        for(int col = 0; col != blocks.cols; ++col)
        {
            cv::Rect const roi(rOffset,col*16,16,16);
            bptr[col] =
                    static_cast<float>(cv::sum(freImg(roi))[0]);
        }
    }
}

void createHash(cv::Mat const &blocks, cv::Mat &hash)
{
    int hash_index = 0;
    int bit_index = 0;
    uchar hashbyte = 0;
    uchar *hashPtr = hash.ptr<uchar>(0);
    for (int row=0; row < 29; row += 4)
    {
        for (int col=0; col < 29; col += 4)
        {
            cv::Rect const roi(col,row,3,3);
            cv::Mat const blockROI = blocks(roi);
            float const avg =
                    static_cast<float>(cv::sum(blockROI)[0]/9.0);
            for(int i = 0; i != blockROI.rows; ++i)
            {
                float const *bptr = blockROI.ptr<float>(i);
                for(int j = 0; j != blockROI.cols; ++j)
                {
                    hashbyte <<= 1;
                    if (bptr[j] > avg)
                    {
                        hashbyte |= 0x01;
                    }
                    ++bit_index;
                    if ((bit_index%8) == 0)
                    {
                        hash_index = (bit_index/8) - 1;
                        hashPtr[hash_index] = hashbyte;
                        hashbyte = 0x00;
                    }
                }
            }
        }
    }
}

class MarrHildrethHashImpl CV_FINAL : public ImgHashBase::ImgHashImpl
{
public:

    MarrHildrethHashImpl(float alpha = 2.0f, float scale = 1.0f) : alphaVal(alpha), scaleVal(scale)
    {
        getMHKernel(alphaVal, scaleVal, mhKernel);
        blocks.create(31,31, CV_32F);
    }

    ~MarrHildrethHashImpl() CV_OVERRIDE { }

    virtual void compute(cv::InputArray inputArr, cv::OutputArray outputArr) CV_OVERRIDE
    {
        cv::Mat const input = inputArr.getMat();
        CV_Assert(input.type() == CV_8UC4 ||
                  input.type() == CV_8UC3 ||
                  input.type() == CV_8U);

        if(input.channels() > 1)
            cv::cvtColor(input, grayImg, COLOR_BGR2GRAY);
        else
            grayImg = input;

        //pHash use Canny-deritch filter to blur the image
        cv::GaussianBlur(grayImg, blurImg, cv::Size(7, 7), 0);
        cv::resize(blurImg, resizeImg, cv::Size(512, 512), 0, 0, INTER_CUBIC);
        cv::equalizeHist(resizeImg, equalizeImg);

        //extract frequency info by mh kernel
        cv::filter2D(equalizeImg, freImg, CV_32F, mhKernel);
        fillBlocks(freImg, blocks);

        outputArr.create(1, 72, CV_8U);
        cv::Mat hash = outputArr.getMat();
        createHash(blocks, hash);
    }

    virtual double compare(cv::InputArray hashOne, cv::InputArray hashTwo) const CV_OVERRIDE
    {
        return norm(hashOne, hashTwo, NORM_HAMMING);
    }

    float getAlpha() const
    {
        return alphaVal;
    }

    float getScale() const
    {
        return scaleVal;
    }

    void setKernelParam(float alpha, float scale)
    {
        alphaVal = alpha;
        scaleVal = scale;
        getMHKernel(alphaVal, scaleVal, mhKernel);
    }

friend class MarrHildrethHash;

private:
    float alphaVal;
    cv::Mat blocks;
    cv::Mat blurImg;
    cv::Mat equalizeImg;
    cv::Mat freImg; //frequency response image
    cv::Mat grayImg;
    cv::Mat mhKernel;
    cv::Mat resizeImg;
    float scaleVal;
};

inline MarrHildrethHashImpl *getLocalImpl(ImgHashBase::ImgHashImpl *ptr)
{
    MarrHildrethHashImpl * impl = static_cast<MarrHildrethHashImpl*>(ptr);
    CV_Assert(impl);
    return impl;
}

}

//==================================================================================================

namespace cv { namespace img_hash {

float MarrHildrethHash::getAlpha() const
{
    return getLocalImpl(pImpl)->getAlpha();
}

float MarrHildrethHash::getScale() const
{
    return getLocalImpl(pImpl)->getScale();
}

void MarrHildrethHash::setKernelParam(float alpha, float scale)
{
    getLocalImpl(pImpl)->setKernelParam(alpha, scale);
}

Ptr<MarrHildrethHash> MarrHildrethHash::create(float alpha, float scale)
{
    Ptr<MarrHildrethHash> res(new MarrHildrethHash);
    res->pImpl = makePtr<MarrHildrethHashImpl>(alpha, scale);
    return res;
}

void marrHildrethHash(cv::InputArray inputArr,
                      cv::OutputArray outputArr,
                      float alpha, float scale)
{
    MarrHildrethHashImpl(alpha, scale).compute(inputArr, outputArr);
}

} } // cv::img_hash::
