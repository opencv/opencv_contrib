// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#include <vector>

namespace opencv_test
{
namespace
{

using namespace std;
/****************************************************************************************\
 *                Test drawing works properly
\****************************************************************************************/

void runCCheckerDraw(Ptr<CChecker> pChecker, int rows, int cols, unsigned int number_of_cells_in_colorchecker)
{
    cv::Mat img(rows, cols, CV_8UC3, {0, 0, 0});

    Ptr<CCheckerDraw> cdraw = CCheckerDraw::create(pChecker);

    cdraw->draw(img);

    //make sure this contains extacly as many rectangles as in the pChecker
    vector<vector<Point>> contours;
    cv::cvtColor(img, img, COLOR_BGR2GRAY);
    findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    ASSERT_EQ(contours.size(), number_of_cells_in_colorchecker);
}

TEST(CV_mccRunCCheckerDrawTest, accuracy_MCC24)
{
    Ptr<CChecker> pChecker = CChecker::create();
    pChecker->setTarget(MCC24);
    pChecker->setBox({{0, 0}, {480, 0}, {480, 640}, {0, 640}});
    runCCheckerDraw(pChecker, 640, 480, 24);
}
TEST(CV_mccRunCCheckerDrawTest, accuracy_SG140)
{
    Ptr<CChecker> pChecker = CChecker::create();
    pChecker->setTarget(SG140);
    pChecker->setBox({{0, 0}, {480, 0}, {480, 640}, {0, 640}});
    runCCheckerDraw(pChecker, 640, 480, 140);
}
TEST(CV_mccRunCCheckerDrawTest, accuracy_VINYL18)
{
    Ptr<CChecker> pChecker = CChecker::create();
    pChecker->setTarget(VINYL18);
    pChecker->setBox({{0, 0}, {480, 0}, {480, 640}, {0, 640}});
    runCCheckerDraw(pChecker, 640, 480, 18);
}

/****************************************************************************************\
 *                Test detection works properly on the simplest images
\****************************************************************************************/

void runCCheckerDetectorBasic(std::string image_name, TYPECHART chartType)
{
    Ptr<CCheckerDetector> detector = CCheckerDetector::create();
    std::string path = cvtest::findDataFile("mcc/" + image_name);
    cv::Mat img = imread(path);
    ASSERT_FALSE(img.empty()) << "Test image can't be loaded: " << path;

    ASSERT_TRUE(detector->process(img, chartType));
}
TEST(CV_mccRunCCheckerDetectorBasic, accuracy_SG140)
{
    runCCheckerDetectorBasic("SG140.png", SG140);
}
TEST(CV_mccRunCCheckerDetectorBasic, accuracy_MCC24)
{
    runCCheckerDetectorBasic("MCC24.png", MCC24);
}

TEST(CV_mccRunCCheckerDetectorBasic, accuracy_VINYL18)
{
    runCCheckerDetectorBasic("VINYL18.png", VINYL18);
}

} // namespace
} // namespace opencv_test
