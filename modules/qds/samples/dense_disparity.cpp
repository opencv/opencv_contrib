#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <opencv2/qds.hpp>




using namespace cv;
using namespace std;


int main()
{

    Mat rightImg, leftImg;

    //Read video meta-data to determine the correct frame size for initialization.
    leftImg = imread("./imgLeft.png", IMREAD_COLOR);
    rightImg = imread("./imgRight.png", IMREAD_COLOR);
    cv::Size frameSize = leftImg.size();
    // Initialize qds and start process.
    qds::QuasiDenseStereo stereo(frameSize);

    uint8_t displvl = 80;					// Number of disparity levels
    cv::Mat disp;

    // Compute dense stereo.
    stereo.process(leftImg, rightImg);

    // Compute disparity between left and right channel of current frame.
    disp = stereo.getDisparity(displvl);

    vector<qds::Match> matches;
    stereo.getDenseMatches(matches);

    // Create three windows and show images.
    cv::namedWindow("right channel");
    cv::namedWindow("left channel");
    cv::namedWindow("disparity map");
    cv::imshow("disparity map", disp);
    cv::imshow("left channel", leftImg);
    cv::imshow("right channel", rightImg);

    std::ofstream dense("./dense.txt", std::ios::out);

    for (uint i=0; i< matches.size(); i++)
    {
        dense << matches[i].p0 << matches[i].p1 << endl;
    }
    dense.close();



    cv::waitKey(0);

    return 0;
}
