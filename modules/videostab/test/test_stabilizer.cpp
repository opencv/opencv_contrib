// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/ts/cuda_test.hpp> // EXPECT_MAT_NEAR

namespace opencv_test { namespace {

using namespace ::cv::videostab;

class OneFrameTestSource : public IFrameSource
{
public:
    OneFrameTestSource(const Mat &frame)
    {
        frameNumber_ = 0;
        frame_ = frame;
    }

    virtual void reset() CV_OVERRIDE
    {
        frameNumber_ = 0;
    }

    virtual Mat nextFrame() CV_OVERRIDE
    {
        return (frameNumber_++ == 0) ? frame_ : Mat();
    }

private:
    int frameNumber_;
    Mat frame_;
};

TEST(OnePassStabilizer, oneFrame)
{
    Mat frame(2, 3, CV_8UC3);
    randu(frame, Scalar::all(0), Scalar::all(255));

    OnePassStabilizer stabilizer;
    stabilizer.setRadius(10);
    stabilizer.setFrameSource(makePtr<OneFrameTestSource>(frame));

    Mat stabilizedFrame = stabilizer.nextFrame();
    EXPECT_MAT_NEAR(frame, stabilizedFrame, 0);
    EXPECT_TRUE(stabilizer.nextFrame().empty());
}

TEST(OnePassStabilizer, oneFrame_deblur)
{
    Mat frame(2, 3, CV_8UC3);
    randu(frame, Scalar::all(0), Scalar::all(255));

    OnePassStabilizer stabilizer;
    stabilizer.setRadius(1);
    stabilizer.setFrameSource(makePtr<OneFrameTestSource>(frame));

    Ptr<WeightingDeblurer> deblurer = makePtr<WeightingDeblurer>();
    deblurer->setRadius(10);
    stabilizer.setDeblurer(deblurer);

    Mat stabilizedFrame = stabilizer.nextFrame();
    EXPECT_MAT_NEAR(frame, stabilizedFrame, 0);
    EXPECT_TRUE(stabilizer.nextFrame().empty());
}

TEST(TwoPassStabilizer, oneFrame)
{
    Mat frame(2, 3, CV_8UC3);
    randu(frame, Scalar::all(0), Scalar::all(255));

    TwoPassStabilizer stabilizer;
    stabilizer.setRadius(10);
    stabilizer.setFrameSource(makePtr<OneFrameTestSource>(frame));

    Mat stabilizedFrame = stabilizer.nextFrame();
    EXPECT_MAT_NEAR(frame, stabilizedFrame, 0);
    EXPECT_TRUE(stabilizer.nextFrame().empty());
}

}} // namespace
