// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

class CV_GMSMatcherTest : public cvtest::BaseTest
{
public:
    CV_GMSMatcherTest();
    ~CV_GMSMatcherTest();

protected:
    virtual void run(int);

    bool combinations[4][2];
    double eps[3][4]; //3 imgs x 4 combinations
    double correctMatchDistThreshold;
};

CV_GMSMatcherTest::CV_GMSMatcherTest()
{
    combinations[0][0] = false; combinations[0][1] = false;
    combinations[1][0] = false; combinations[1][1] = true;
    combinations[2][0] = true; combinations[2][1] = false;
    combinations[3][0] = true; combinations[3][1] = true;

    //Threshold = truncate(min(acc_win32, acc_win64))
    eps[0][0] = 0.9313;
    eps[0][1] = 0.9223;
    eps[0][2] = 0.9313;
    eps[0][3] = 0.9223;

    eps[1][0] = 0.8199;
    eps[1][1] = 0.7964;
    eps[1][2] = 0.8199;
    eps[1][3] = 0.7964;

    eps[2][0] = 0.7098;
    eps[2][1] = 0.6659;
    eps[2][2] = 0.6939;
    eps[2][3] = 0.6457;

    correctMatchDistThreshold = 5.0;
}

CV_GMSMatcherTest::~CV_GMSMatcherTest() {}

void CV_GMSMatcherTest::run( int )
{
    ts->set_failed_test_info(cvtest::TS::OK);

    Mat imgRef = imread(string(ts->get_data_path()) + "detectors_descriptors_evaluation/images_datasets/graf/img1.png");

    Ptr<Feature2D> orb = ORB::create(10000);
    vector<KeyPoint> keypointsRef, keypointsCur;
    Mat descriptorsRef, descriptorsCur;
    orb->detectAndCompute(imgRef, noArray(), keypointsRef, descriptorsRef);

    vector<DMatch> matchesAll, matchesGMS;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    const int startImg = 2;
    const int nImgs = 3;
    for (int num = startImg; num < startImg+nImgs; num++)
    {
        string imgPath = string(ts->get_data_path()) + format("detectors_descriptors_evaluation/images_datasets/graf/img%d.png", num);
        Mat imgCur = imread(imgPath);
        orb->detectAndCompute(imgCur, noArray(), keypointsCur, descriptorsCur);

        matcher->match(descriptorsCur, descriptorsRef, matchesAll);

        string xml = string(ts->get_data_path()) + format("detectors_descriptors_evaluation/images_datasets/graf/H1to%dp.xml", num);
        FileStorage fs(xml, FileStorage::READ);
        if (!fs.isOpened())
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }

        Mat H1toCur;
        fs[format("H1%d", num)] >> H1toCur;

        for (int comb = 0; comb < 4; comb++)
        {
            matchGMS(imgCur.size(), imgRef.size(), keypointsCur, keypointsRef, matchesAll, matchesGMS, combinations[comb][0], combinations[comb][1]);

            int nbCorrectMatches = 0;
            for (size_t i = 0; i < matchesGMS.size(); i++)
            {
                Point2f ptRef = keypointsRef[matchesGMS[i].trainIdx].pt;
                Point2f ptCur = keypointsCur[matchesGMS[i].queryIdx].pt;
                Mat matRef = (Mat_<double>(3,1) << ptRef.x, ptRef.y, 1);
                Mat matTrans = H1toCur * matRef;
                Point2f ptTrans( (float) (matTrans.at<double>(0,0)/matTrans.at<double>(2,0)),
                                 (float) (matTrans.at<double>(1,0)/matTrans.at<double>(2,0)));

                if (cv::norm(ptTrans-ptCur) < correctMatchDistThreshold)
                    nbCorrectMatches++;
            }

            double ratio = nbCorrectMatches / (double) matchesGMS.size();
            if (ratio < eps[num-startImg][comb])
            {
                ts->printf( cvtest::TS::LOG, "Invalid accuracy for image %s and combination withRotation=%d withScale=%d, "
                                             "matches ratio is %f, ratio threshold is %f, distance threshold is %f.\n",
                            imgPath.substr(imgPath.size()-8).c_str(), combinations[comb][0], combinations[comb][1], ratio,
                            eps[num-startImg][comb], correctMatchDistThreshold);
                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            }
        }
    }
}

TEST(XFeatures2d_GMSMatcher, gms_matcher_regression) { CV_GMSMatcherTest test; test.safe_run(); }

}} // namespace
