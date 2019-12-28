// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {


const string INPUT_DIR = "cv/ximgproc/globalmatting";

TEST(GlobalMattingTest, accuracy)
{
    Ptr<GlobalMatting> gm = createGlobalMatting();
    string img_path = cvtest::findDataFile(INPUT_DIR + "/input.png");
    string trimap_path = cvtest::findDataFile(INPUT_DIR + "/trimap.png");

    Mat img     = imread(img_path, IMREAD_COLOR);
    Mat trimap  = imread(trimap_path, IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty()) << "The Image could not be loaded: "<< img_path;
    ASSERT_FALSE(trimap.empty()) << "The trimap could not be loaded: "<< trimap_path;

    ASSERT_EQ(img.cols, trimap.cols);
    ASSERT_EQ(img.rows, trimap.rows);

    Mat foreground, alpha;
    int niter = 9;
    gm->getMat(img, trimap, foreground, alpha, niter);

    ASSERT_FALSE(foreground.empty()) << " Could not extract the foreground ";
    ASSERT_FALSE(alpha.empty()) << " Could not generate alpha matte ";

    ASSERT_EQ(img.cols, alpha.cols);
    ASSERT_EQ(img.rows, alpha.rows);

    std::string ref_alpha_path = cvtest::findDataFile(INPUT_DIR + "/ref_alpha.png");
    Mat ref_alpha = imread(trimap_path, IMREAD_GRAYSCALE);

    EXPECT_LE(cvtest::norm(ref_alpha, alpha, NORM_L2 | NORM_RELATIVE), 1e-3);  // FIXIT! Result is unstable
#if 0
    imshow("alpha", alpha);
    waitKey();
    imwrite("globalmatting_alpha.png", alpha);
#endif
}


}}  // namespace
