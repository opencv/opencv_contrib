#include "opencv2/xphoto.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

const char *keys = { "{help h usage ? |         | print this message}"
                     "{i              |         | input image name  }"
                     "{o              |         | output image name }"
                     "{a              |grayworld| color balance algorithm (simple, grayworld or learning_based)}"
                     "{m              |         | path to the model for the learning-based algorithm (optional) }" };

int main(int argc, const char **argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("OpenCV color balance demonstration sample");
    if (parser.has("help") || argc < 2)
    {
        parser.printMessage();
        return 0;
    }

    string inFilename = parser.get<string>("i");
    string outFilename = parser.get<string>("o");
    string algorithm = parser.get<string>("a");
    string modelFilename = parser.get<string>("m");

    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }

    Mat src = imread(inFilename, 1);
    if (src.empty())
    {
        printf("Cannot read image file: %s\n", inFilename.c_str());
        return -1;
    }

    Mat res;
    Ptr<xphoto::WhiteBalancer> wb;
    if (algorithm == "simple")
        wb = xphoto::createSimpleWB();
    else if (algorithm == "grayworld")
        wb = xphoto::createGrayworldWB();
    else if (algorithm == "learning_based")
        wb = xphoto::createLearningBasedWB(modelFilename);
    else
    {
        printf("Unsupported algorithm: %s\n", algorithm.c_str());
        return -1;
    }

    wb->balanceWhite(src, res);

    if (outFilename == "")
    {
        namedWindow("after white balance", 1);
        imshow("after white balance", res);

        waitKey(0);
    }
    else
        imwrite(outFilename, res);

    return 0;
}
