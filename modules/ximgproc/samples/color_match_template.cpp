#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/ximgproc/color_match.hpp>

using namespace std;
using namespace cv;



static void AddSlider(String sliderName, String windowName, int minSlider, int maxSlider, int valDefault, int *valSlider, void(*f)(int, void *), void *r)
{
    createTrackbar(sliderName, windowName, valSlider, 1, f, r);
    setTrackbarMin(sliderName, windowName, minSlider);
    setTrackbarMax(sliderName, windowName, maxSlider);
    setTrackbarPos(sliderName, windowName, valDefault);
}

struct SliderData {
    Mat img;
    int thresh;
};

static void UpdateThreshImage(int , void *r)
{
    SliderData *p = (SliderData*)r;
    Mat dst,labels,stats,centroids;

    threshold(p->img, dst, p->thresh, 255, THRESH_BINARY);

    connectedComponentsWithStats(dst, labels, stats, centroids, 8);
    if (centroids.rows < 10)
    {
        cout << "**********************************************************************************\n";
        for (int i = 0; i < centroids.rows; i++)
        {
            cout << dst.cols - centroids.at<double>(i, 0)  << " ";
            cout << dst.rows -  centroids.at<double>(i, 1)  << "\n";
        }
        cout << "----------------------------------------------------------------------------------\n";
    }
    flip(dst, dst, -1);

    imshow("Max Quaternion corr",dst);
}

int main(int argc, char *argv[])
{
    cv::CommandLineParser parser(argc, argv,
        "{help h | | match color image }{@colortemplate | | input color template image}{@colorimage | | input color image}");
    if (parser.has("help"))
    {
        parser.printMessage();
        return -1;
    }
    string templateName = parser.get<string>("@colortemplate");
    if (templateName.empty())
    {
        parser.printMessage();
        parser.printErrors();
        return -2;
    }
    string colorImageName = parser.get<string>("@colorimage");
    if (templateName.empty())
    {
        parser.printMessage();
        parser.printErrors();
        return -2;
    }
    Mat imgLogo = imread(templateName, IMREAD_COLOR);
    Mat imgColor = imread(colorImageName, IMREAD_COLOR);
    imshow("Image", imgColor);
    imshow("template", imgLogo);
    // OK NOW WHERE IS OPENCV LOGO ?
    Mat imgcorr;
    SliderData ps;
    ximgproc::colorMatchTemplate(imgColor, imgLogo, imgcorr);
    imshow("quaternion correlation real", imgcorr);
    normalize(imgcorr, imgcorr,1,0,NORM_MINMAX);
    imgcorr.convertTo(ps.img, CV_8U, 255);
    imshow("quaternion correlation", imgcorr);
    ps.thresh = 0;
    AddSlider("Level", "quaternion correlation", 0, 255, ps.thresh, &ps.thresh, UpdateThreshImage, &ps);
    int code = 0;
    while (code != 27)
    {
        code = waitKey(50);
    }

    waitKey(0);
    return 0;
}