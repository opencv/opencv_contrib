// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

CV_TEST_MAIN("cv")

namespace opencv_test { namespace {

TEST(CV_Rapid, rapid)
{
    // a unit sized box
    std::vector<Vec3f> vtx = {
        {1, -1, -1}, {1, -1, 1}, {-1, -1, 1}, {-1, -1, -1}, {1, 1, -1}, {1, 1, 1}, {-1, 1, 1}, {-1, 1, -1},
    };
    std::vector<Vec3i> tris = {
        {2, 4, 1}, {8, 6, 5}, {5, 2, 1}, {6, 3, 2}, {3, 8, 4}, {1, 8, 5},
        {2, 3, 4}, {8, 7, 6}, {5, 6, 2}, {6, 7, 3}, {3, 7, 8}, {1, 4, 8},
    };
    Mat(tris) -= Scalar(1, 1, 1);

    // camera setup
    Size sz(1280, 720);

    Mat K = getDefaultNewCameraMatrix(Matx33f::diag(Vec3f(800, 800, 1)), sz, true);
    Vec3f trans = {0, 0, 5};
    Vec3f rot = {0.7f, 0.6f, 0};

    // draw something
    Mat pts2d;
    projectPoints(vtx, rot, trans, K, noArray(), pts2d);

    Mat_<uchar> img(sz, uchar(0));
    rapid::drawWireframe(img, pts2d, tris, Scalar(255), LINE_8);

    // recover pose form different position
    Vec3f t_init = Vec3f(0.1f, 0, 5);
    auto tracker = rapid::Rapid::create(vtx, tris);
    // do two iterations
    TermCriteria term(TermCriteria::MAX_ITER, 2, 0);
    tracker->compute(img, 100, 20, K, rot, t_init, term);

    // assert that it improved from init
    ASSERT_LT(cv::norm(trans - t_init), 0.075);
}

}} // namespace
