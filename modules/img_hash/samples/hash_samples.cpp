#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/img_hash.hpp"

#include <iostream>

using namespace cv;
using namespace cv::img_hash;
using namespace std;

template <typename T>
inline void test_one(const std::string &title, const Mat &a, const Mat &b)
{
    cout << "=== " << title << " ===" << endl;
    TickMeter tick;
    Mat hashA, hashB;
    Ptr<ImgHashBase> func;
    func = T::create();

    tick.reset(); tick.start();
    func->compute(a, hashA);
    tick.stop();
    cout << "compute1: " << tick.getTimeMilli() << " ms" << endl;

    tick.reset(); tick.start();
    func->compute(b, hashB);
    tick.stop();
    cout << "compute2: " << tick.getTimeMilli() << " ms" << endl;

    cout << "compare: " << func->compare(hashA, hashB) << endl << endl;;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cerr << "must input the path of input image and target image. ex : hash_samples lena.jpg lena2.jpg" << endl;
        return -1;
    }
    ocl::setUseOpenCL(false);

    Mat input = imread(argv[1]);
    Mat target = imread(argv[2]);

    test_one<AverageHash>("AverageHash", input, target);
    test_one<PHash>("PHash", input, target);
    test_one<MarrHildrethHash>("MarrHildrethHash", input, target);
    test_one<RadialVarianceHash>("RadialVarianceHash", input, target);
    test_one<BlockMeanHash>("BlockMeanHash", input, target);

    return 0;
}
