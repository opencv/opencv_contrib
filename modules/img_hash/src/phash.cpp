// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

using namespace cv;
using namespace cv::img_hash;
using namespace std;

namespace {

class PHashImpl CV_FINAL : public ImgHashBase::ImgHashImpl
{
public:
    virtual void compute(cv::InputArray inputArr, cv::OutputArray outputArr) CV_OVERRIDE
    {
        cv::Mat const input = inputArr.getMat();
        CV_Assert(input.type() == CV_8UC4 ||
                  input.type() == CV_8UC3 ||
                  input.type() == CV_8U);

        cv::resize(input, resizeImg, cv::Size(32,32), 0, 0, INTER_LINEAR_EXACT);
        if(input.type() == CV_8UC3)
        {
            cv::cvtColor(resizeImg, grayImg, CV_BGR2GRAY);
        }
        else if(input.type() == CV_8UC4)
        {
            cv::cvtColor(resizeImg, grayImg, CV_BGRA2GRAY);
        }
        else
        {
            grayImg = resizeImg;
        }

        grayImg.convertTo(grayFImg, CV_32F);
        cv::dct(grayFImg, dctImg);
        dctImg(cv::Rect(0, 0, 8, 8)).copyTo(topLeftDCT);
        topLeftDCT.at<float>(0, 0) = 0;
        float const imgMean = static_cast<float>(cv::mean(topLeftDCT)[0]);

        cv::compare(topLeftDCT, imgMean, bitsImg, CMP_GT);
        bitsImg /= 255;
        outputArr.create(1, 8, CV_8U);
        cv::Mat hash = outputArr.getMat();
        uchar *hash_ptr = hash.ptr<uchar>(0);
        uchar const *bits_ptr = bitsImg.ptr<uchar>(0);
        std::bitset<8> bits;
        for(size_t i = 0, j = 0; i != bitsImg.total(); ++j)
        {
            for(size_t k = 0; k != 8; ++k)
            {
                //avoid warning C4800, casting do not work
                bits[k] = bits_ptr[i++] != 0;
            }
            hash_ptr[j] = static_cast<uchar>(bits.to_ulong());
        }
    }

    virtual double compare(cv::InputArray hashOne, cv::InputArray hashTwo) const CV_OVERRIDE
    {
        return norm(hashOne, hashTwo, NORM_HAMMING);
    }

private:
    cv::Mat bitsImg;
    cv::Mat dctImg;
    cv::Mat grayFImg;
    cv::Mat grayImg;
    cv::Mat resizeImg;
    cv::Mat topLeftDCT;
};

} // namespace::

//==================================================================================================

namespace cv { namespace img_hash {

Ptr<PHash> PHash::create()
{
    Ptr<PHash> res(new PHash);
    res->pImpl = makePtr<PHashImpl>();
    return res;
}

void pHash(cv::InputArray inputArr, cv::OutputArray outputArr)
{
    PHashImpl().compute(inputArr, outputArr);
}

} } // cv::img_hash::
