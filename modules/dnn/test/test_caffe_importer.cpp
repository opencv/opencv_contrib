#include "test_precomp.hpp"

namespace cvtest
{

using namespace std;
using namespace std::tr1;
using namespace testing;
using namespace cv;
using namespace cv::dnn;

static std::string getOpenCVExtraDir()
{
    return cvtest::TS::ptr()->get_data_path();
}

template<typename TStr>
static std::string getTestFile(TStr filename)
{
    return (getOpenCVExtraDir() + "/dnn/") + filename;
}

TEST(ReadCaffe_GTSRB, Accuracy)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(getTestFile("gtsrb.prototxt"), "");
        importer->populateNet(net);
    }

//    Mat img = imread(getTestFile("sign_50.ppm"));
//    CV_Assert(!img.empty());
//    img.convertTo(img, CV_32F, 1.0 / 255);
//    resize(img, img, cv::Size(48, 48));
//    Blob imgBlob(img);

//    net.setBlob("input", imgBlob);
//    net.forward();

//    Blob res = net.getBlob("loss");
//    for (int n = 0; n < 1; n++)
//    {
//        Mat slice = Mat(res.channels() * res.rows(), res.cols(), CV_32F, res.ptr<float>(n));

//        double maxv;
//        std::vector<int> maxIdx;
//        minMaxLoc(slice, NULL, &maxv, NULL, &maxIdx);

//        int bestClass = maxIdx[0];
//        std::cout << "Best class: #" <<  bestClass << std::endl;

//        //imwrite(getTestFile("vis.png"), slice*(255.0 / maxv));
//    }
}

TEST(ReadCaffe_GoogleNet, Accuracy)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(getTestFile("googlenet_deploy.prototxt"), "");
        importer->populateNet(net);
    }
}

}
