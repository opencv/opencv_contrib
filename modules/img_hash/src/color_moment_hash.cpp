// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

using namespace cv;
using namespace cv::img_hash;
using namespace std;

namespace {

class ColorMomentHashImpl : public ImgHashBase::ImgHashImpl
{
public:
    ~ColorMomentHashImpl() {}

    virtual void compute(cv::InputArray inputArr, cv::OutputArray outputArr)
    {
      cv::Mat const input = inputArr.getMat();
      CV_Assert(input.type() == CV_8UC4 ||
                input.type() == CV_8UC3 ||
                input.type() == CV_8U);

      if(input.type() == CV_8UC3)
      {
          colorImg_ = input;
      }
      else if(input.type() == CV_8UC4)
      {
          cv::cvtColor(input, colorImg_, CV_BGRA2BGR);
      }
      else
      {
          cv::cvtColor(input, colorImg_, CV_GRAY2BGR);
      }

      cv::resize(colorImg_, resizeImg_, cv::Size(512,512), 0, 0,
                 INTER_CUBIC);
      cv::GaussianBlur(resizeImg_, blurImg_, cv::Size(3,3), 0, 0);

      cv::cvtColor(blurImg_, colorSpace_, CV_BGR2HSV);
      cv::split(colorSpace_, channels_);
      outputArr.create(1, 42, CV_64F);
      cv::Mat hash = outputArr.getMat();
      hash.setTo(0);
      computeMoments(hash.ptr<double>(0));

      cv::cvtColor(blurImg_, colorSpace_, CV_BGR2YCrCb);
      cv::split(colorSpace_, channels_);
      computeMoments(hash.ptr<double>(0) + 21);
    }

    virtual double compare(cv::InputArray hashOne, cv::InputArray hashTwo) const
    {
      return norm(hashOne, hashTwo, NORM_L2) * 10000;
    }

private:
    void computeMoments(double *inout)
    {
      for(size_t i = 0; i != channels_.size(); ++i)
      {
        cv::HuMoments(cv::moments(channels_[i]), inout);
        inout += 7;
      }
    }

private:
    cv::Mat blurImg_;
    cv::Mat colorImg_;
    std::vector<cv::Mat> channels_;
    cv::Mat colorSpace_;
    cv::Mat resizeImg_;
};

}

//==================================================================================================

namespace cv { namespace img_hash {

Ptr<ColorMomentHash> ColorMomentHash::create()
{
    Ptr<ColorMomentHash> res(new ColorMomentHash);
    res->pImpl = makePtr<ColorMomentHashImpl>();
    return res;
}

void colorMomentHash(cv::InputArray inputArr, cv::OutputArray outputArr)
{
    ColorMomentHashImpl().compute(inputArr, outputArr);
}

} } // cv::img_hash::
