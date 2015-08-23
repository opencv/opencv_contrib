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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#if defined(ENABLE_TORCH_IMPORTER) && ENABLE_TORCH_IMPORTER
#if defined(ENABLE_TORCH_TESTS) && ENABLE_TORCH_TESTS
#include "test_precomp.hpp"

namespace cvtest
{

using namespace std;
using namespace testing;
using namespace cv;
using namespace cv::dnn;

template<typename TStr>
static std::string _tf(TStr filename)
{
    return (getOpenCVExtraDir() + "/dnn/torch/") + filename;
}

TEST(Torch_Importer, simple_read)
{
    Net net;
    Ptr<Importer> importer;

    ASSERT_NO_THROW( importer = createTorchImporter(_tf("net_simple_net.txt"), false) );
    ASSERT_TRUE( importer != NULL );
    importer->populateNet(net);
}

static void runTorchNet(String prefix, String outLayerName, bool isBinary)
{
    String suffix = (isBinary) ? ".dat" : ".txt";

    Net net;
    Ptr<Importer> importer = createTorchImporter(_tf(prefix + "_net" + suffix), isBinary);
    ASSERT_TRUE(importer != NULL);
    importer->populateNet(net);

    Blob inp, outRef;
    ASSERT_NO_THROW( inp = readTorchBlob(_tf(prefix + "_input" + suffix), isBinary) );
    ASSERT_NO_THROW( outRef = readTorchBlob(_tf(prefix + "_output" + suffix), isBinary) );

    net.setBlob(".0", inp);
    net.forward();
    Blob out = net.getBlob(outLayerName);

    normAssert(outRef, out);
}

TEST(Torch_Importer, run_convolution)
{
    runTorchNet("net_conv", "l1_Convolution", false);
}

TEST(Torch_Importer, run_pool_max)
{
    runTorchNet("net_pool_max", "l1_Pooling", false);
}

TEST(Torch_Importer, run_pool_ave)
{
    runTorchNet("net_pool_ave", "l1_Pooling", false);
}

TEST(Torch_Importer, run_reshape)
{
    runTorchNet("net_reshape", "l1_Reshape", false);
    runTorchNet("net_reshape_batch", "l1_Reshape", false);
}

TEST(Torch_Importer, run_linear)
{
    runTorchNet("net_linear_2d", "l1_InnerProduct", false);
}

TEST(Torch_Importer, run_paralel)
{
    runTorchNet("net_parallel", "l2_torchMerge", false);
}

TEST(Torch_Importer, run_concat)
{
    runTorchNet("net_concat", "l2_torchMerge", false);
}

}
#endif
#endif
