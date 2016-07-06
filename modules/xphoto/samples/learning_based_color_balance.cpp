#include "opencv2/xphoto.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

const char *keys = {"{help h usage ? |      | print this message}"
                    "{i              |      | input image name  }"
                    "{o              |      | output image name }"};

int main(int argc, const char **argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("OpenCV learning-based color balance demonstration sample");
    if (parser.has("help") || argc < 2)
    {
        parser.printMessage();
        return 0;
    }

    string inFilename = parser.get<string>("i");
    string outFilename = parser.get<string>("o");

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
    xphoto::autowbLearningBased(src, res);

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
