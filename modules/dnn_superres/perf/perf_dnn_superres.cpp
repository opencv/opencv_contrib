// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;

namespace opencv_test { namespace {

typedef perf::TestBaseWithParam<tuple<tuple<string,string,int>, string> > dnn_superres;

// lapsrn inference fails after conversion to ONNX
// BUG https://github.com/opencv/opencv_contrib/issues/4263
#define MODEL testing::Values(tuple<string,string,int> {"espcn","cv/dnn_superres/ESPCN_x2.onnx",2})
                           //, tuple<string,string,int> {"lapsrn","dnn/onnx/models/LapSRN_x4.onnx",4})
#define IMAGES testing::Values("cv/dnn_superres/butterfly.png", "cv/shared/baboon.png", "cv/shared/lena.png")

PERF_TEST_P(dnn_superres, upsample, testing::Combine(MODEL, IMAGES))
{
    tuple<string,string,int> model = get<0>( GetParam() );
    string image_name = get<1>( GetParam() );

    string model_name = get<0>(model);
    string model_filename = get<1>(model);
    int scale = get<2>(model);

    string model_path = cvtest::findDataFile(model_filename);
    string image_path = cvtest::findDataFile(image_name);

    DnnSuperResImpl sr;
    sr.readModel(model_path);
    sr.setModel(model_name, scale);

    Mat img = imread(image_path);
    ASSERT_FALSE(img.empty()) << image_path;

    Mat result;

    TEST_CYCLE() { sr.upsample(img, result); }

    ASSERT_FALSE(result.empty());

    SANITY_CHECK_NOTHING();
}

}}
