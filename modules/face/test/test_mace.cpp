// This file is part of the OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <fstream>

namespace opencv_test { namespace {

const string FACE_DIR = "face";
const int WINDOW_SIZE = 64;

class MaceTest
{
public:

    MaceTest(bool salt);
    void run();

protected:
    Ptr<MACE> mace;
    bool salt;
};

MaceTest::MaceTest(bool use_salt)
{
    mace = MACE::create(WINDOW_SIZE);
    salt = use_salt;
}

void MaceTest::run()
{
    Rect david1 (125,66,58,56);
    Rect david2 (132,69,73,74);
    Rect detect (199,124,256,274);
    string folder = cvtest::TS::ptr()->get_data_path() + FACE_DIR;
    Mat train = imread(folder + "/david2.jpg", 0);
    Mat tst_p = imread(folder + "/david1.jpg", 0);
    Mat tst_n = imread(folder + "/detect.jpg", 0);
    vector<Mat> sam_train;
    sam_train.push_back( train(Rect(132,69,73,74)) );
    sam_train.push_back( train(Rect(130,69,73,72)) );
    sam_train.push_back( train(Rect(134,67,73,74)) );
    sam_train.push_back( tst_p(Rect(125,66,58,56)) );
    sam_train.push_back( tst_p(Rect(123,67,55,58)) );
    sam_train.push_back( tst_p(Rect(125,65,58,60)) );

    if (salt) mace->salt("it's david"); // "owner's" salt
    mace->train(sam_train);
    bool self_ok = mace->same(train(david2));
    if (salt) mace->salt("this is a test"); // "other's" salt
    bool false_A = mace->same(tst_n(detect));
    ASSERT_TRUE(self_ok);
    ASSERT_FALSE(false_A);
}


TEST(MACE_, unsalted)
{
    MaceTest test(false); test.run();
}
TEST(MACE_, salted)
{
    MaceTest test(true); test.run();
}


}} // namespace
