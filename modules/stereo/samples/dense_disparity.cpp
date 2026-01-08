#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <opencv2/stereo.hpp>




using namespace cv;
using namespace std;


int main()
{
//!     [load]
    cv::Mat rightImg, leftImg;
    leftImg = imread("./imgLeft.png", IMREAD_COLOR);
    rightImg = imread("./imgRight.png", IMREAD_COLOR);
//!     [load]


//!     [create]
    cv::Size frameSize = leftImg.size();
    Ptr<stereo::QuasiDenseStereo> stereo = stereo::QuasiDenseStereo::create(frameSize);
//!     [create]


//!     [process]
    stereo->process(leftImg, rightImg);
//!     [process]


//!     [disp]
    cv::Mat disp;
    disp = stereo->getDisparity();
    cv::namedWindow("disparity map");
    cv::imshow("disparity map", disp);
//!     [disp]


    cv::namedWindow("right channel");
    cv::namedWindow("left channel");
    cv::imshow("left channel", leftImg);
    cv::imshow("right channel", rightImg);


//!     [export]
    vector<stereo::MatchQuasiDense> matches;
    stereo->getDenseMatches(matches);
    std::ofstream dense("./dense.txt", std::ios::out);
    for (uint i=0; i< matches.size(); i++)
    {
        dense << matches[i].p0 << matches[i].p1 << endl;
    }
    dense.close();
//!     [export]



    cv::waitKey(0);

    return 0;
}
