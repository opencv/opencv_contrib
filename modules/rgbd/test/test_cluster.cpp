/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2015, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"
#include <opencv2/rgbd.hpp>

namespace cv
{
namespace rgbd
{

class CV_EXPORTS TickMeter
{
public:
    TickMeter();
    void start();
    void stop();

    int64 getTimeTicks() const;
    double getTimeMicro() const;
    double getTimeMilli() const;
    double getTimeSec()   const;
    int64 getCounter() const;

    void reset();
private:
    int64 counter;
    int64 sumTime;
    int64 startTime;
};

class CV_RgbdClusterTest : public cvtest::BaseTest
{
public:
    CV_RgbdClusterTest()
    {
    }
    ~CV_RgbdClusterTest()
    {
    }
protected:
    void
        run(int)
    {
            try
            {
                RgbdCluster rgbdCluster;

                // load test data

                for (int ii = 0; ii < 10; ii++)
                {
                    // test performance
                    testit();
                }
            }
            catch (...)
            {
                ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
            }
            ts->set_failed_test_info(cvtest::TS::OK);
        }

    void
        testit()
    {
            for (char i_test = 0; i_test < 2; ++i_test)
            {
                TickMeter tm1, tm2;

                if (i_test == 0)
                {
                    tm1.start();
                    // planar segmentation
                    tm1.stop();

                    tm2.start();
                    // clustering
                    tm2.stop();
                }
                else
                {
                    tm2.start();
                    // clustering
                    tm2.stop();
                }

                // test number of clusters
                //ASSERT_EQ(obtained_clusters, ground_truth);

                std::cout << " Speed: ";
                if (i_test == 0)
                    std::cout << "planar " << tm1.getTimeMilli() << " ms and ";
                std::cout << "cluster " << tm2.getTimeMilli() << " ms " << std::endl;
            }
        }
};

}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


TEST(Rgbd_Cluster, compute)
{
    cv::rgbd::CV_RgbdClusterTest test;
    test.safe_run();
}