// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;

namespace opencv_test { namespace {

typedef perf::TestBaseWithParam<tuple<tuple<string,string,int>, string> > dnn_superres;

#define MODEL testing::Values(tuple<string,string,int> {"espcn","ESPCN_x2.pb",2}, \
                                tuple<string,string,int> {"lapsrn","LapSRN_x4.pb",4})
#define IMAGES testing::Values("cv/dnn_superres/butterfly.png", "cv/shared/baboon.png", "cv/shared/lena.png")

const string TEST_DIR = "cv/dnn_superres";

PERF_TEST_P(dnn_superres, upsample, testing::Combine(MODEL, IMAGES))
{
    tuple<string,string,int> model = get<0>( GetParam() );
    string image_name = get<1>( GetParam() );

    string model_name = get<0>(model);
    string model_filename = get<1>(model);
    int scale = get<2>(model);

    string model_path = getDataPath( TEST_DIR + "/" + model_filename );
    string image_path = getDataPath( image_name );

    DnnSuperResImpl sr;
    sr.readModel(model_path);
    sr.setModel(model_name, scale);

    Mat img = imread(image_path);

    Mat img_new(img.rows * scale, img.cols * scale, CV_8UC3);

    declare.in(img, WARMUP_RNG).out(img_new).iterations(10);

    TEST_CYCLE() { sr.upsample(img, img_new); }

    SANITY_CHECK_NOTHING();
}

}}