// This file is part of the OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <fstream>

namespace opencv_test { namespace {

//
// train on one person, and test against the other
//
#define TESTSET_NAMES testing::Values("david","dudek")

const string TRACKING_DIR = "tracking";
const string FOLDER_IMG = "data";


class MaceTest
{
public:

    MaceTest(string _video, bool salt);
    void run();

protected:
    vector<Rect> boxes(const string &fn);
    vector<Mat> samples(const string &name, int N,int off=0);
    int found(const string &vid);

    Ptr<MACE> mace;

    string video; // train
    string vidA; // test

    int nSampsTest;
    int nSampsTrain;
    int nStep;
    bool salt;
};

MaceTest::MaceTest(string _video, bool use_salt)
{
    int Z = 64; // window size
    mace = MACE::create(Z);

    video = _video;
    if (video=="david") { vidA="dudek"; }
    if (video=="dudek") { vidA="david"; }

    nStep = 2;
    nSampsTest = 5;
    nSampsTrain = 35;
    salt = use_salt;
}

vector<Rect> MaceTest::boxes(const string &fn)
{
    std::ifstream in(fn.c_str());
    int x,y,w,h;
    char sep;
    vector<Rect> _boxes;
    while (in.good() && (in >> x >> sep >> y >> sep >> w >> sep >> h))
    {
        _boxes.push_back( Rect(x,y,w,h) );
    }
    return _boxes;
}

void MaceTest::run()
{
    vector<Mat> sam_train = samples(video, nSampsTrain, 0);
    if (salt) mace->salt(video); // "owner's" salt with "two factor"
    mace->train(sam_train);
    int self_ok = found(video);
    if (salt) mace->salt(vidA); // "other's" salt
    int false_A = found(vidA);
    ASSERT_GE(self_ok, nSampsTest/2);  // it may miss positives
    ASSERT_EQ(false_A, 0);  // but *absolutely* no false positives allowed.
}

int MaceTest::found(const string &vid)
{
    vector<Mat> sam_test = samples(vid, nSampsTest, (1+nStep*nSampsTrain));
    int hits = 0;
    for (size_t i=0; i<sam_test.size(); i++)
    {
        hits += mace->same(sam_test[i]);
    }
    return hits;
}

vector<Mat> MaceTest::samples(const string &name, int N, int off)
{
    string folder = cvtest::TS::ptr()->get_data_path() + TRACKING_DIR + "/" + name;
    string vid  = folder + "/" + FOLDER_IMG + "/" + name + ".webm";
    string anno = folder + "/gt.txt";
    vector<Rect> bb = boxes(anno);
    int startFrame = (name=="david") ? 300 : 0;
    VideoCapture c;
    EXPECT_TRUE(c.open(vid));
    vector<Mat> samps;
    while (samps.size() < size_t(N))
    {
        int frameNo = startFrame + off;
        c.set(CAP_PROP_POS_FRAMES, frameNo);
        Mat frame;
        c >> frame;
        Rect r = bb[off];
        off += nStep;
        samps.push_back(frame(r));
    }
    c.release();
    return samps;
}

//[TESTDATA]
PARAM_TEST_CASE(MACE_, string)
{
    string dataset;
    virtual void SetUp()
    {
        dataset = GET_PARAM(0);
    }
};


TEST_P(MACE_, unsalted)
{
    MaceTest test(dataset, false); test.run();
}
TEST_P(MACE_, salted)
{
    MaceTest test(dataset, true); test.run();
}


INSTANTIATE_TEST_CASE_P(Face, MACE_, TESTSET_NAMES);

}} // namespace
