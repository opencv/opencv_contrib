#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/haze_removal.hpp"

#include <iostream>
using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cerr << "must input the path of input image. ex : ./haze_removal_sample input.jpg" << endl;
        return -1;
    }
    Mat input, output;
    input = imread(argv[1]);
    namedWindow("original", WINDOW_AUTOSIZE);
    imshow("original", input);

    Ptr<cv::haze_removal::DarkChannelPriorHazeRemoval> dehazer = cv::haze_removal::DarkChannelPriorHazeRemoval::create();
    dehazer->setKernel(15, cv::MORPH_RECT);
    dehazer->dehaze(input, output);

    namedWindow("radiance", WINDOW_AUTOSIZE);
    imshow("radiance", output);
    waitKey(0);
    return 0;
}
