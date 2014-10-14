#include <opencv2/photoeffects.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <iostream>

using namespace cv;
using namespace cv::photoeffects;
using namespace std;

const char *helper =
"./tint_sample <img>\n\
\t<img> - file name contained the processed image\n\
";
const char *nameWinImage = "Image";
const char *nameWinFilter = "Filter";
const char *nameWinParam = "Settings";

Vec3b ColorTint;
Mat BaseColor, img, filterImg;
int valueHue = 0;
int valueDen = 50;

void preparePicture();
void trackbarTint(int, void *);
void trackbarDen(int, void *);
int processArguments(int argc, char** argv, Mat &image);

int main(int argc, char** argv)
{
    if (processArguments(argc, argv, img) != 0)
    {
        cout << helper << endl;
        return 1;
    }

    namedWindow(nameWinParam);
    createTrackbar("Hue", nameWinParam, &valueHue, 360, trackbarTint);
    createTrackbar("Density(%)", nameWinParam, &valueDen, 100, trackbarDen);
    preparePicture();

    namedWindow(nameWinImage);
    namedWindow(nameWinFilter);

    imshow(nameWinParam, BaseColor);
    imshow(nameWinImage, img);
    imshow(nameWinFilter, img);

    waitKey();
    destroyAllWindows();
    return 0;
}

int processArguments(int argc, char **argv, Mat &image)
{
    if (argc < 2)
    {
        return 1;
    }
    image = imread(argv[1], 1);
    return 0;
}

void preparePicture()
{
    BaseColor.create(20, 360, CV_8UC3);
    Vec3b hsv;
    for (int j = 0; j < 360; j++)
    {
        hsv[0] = (uchar)((j + 1) / 2);
        hsv[1] = 255;
        hsv[2] = 255;
        for (int i = 0; i < 20; i++)
        {
            BaseColor.at<Vec3b>(i,j) = hsv;
        }
    }
    cvtColor(BaseColor, BaseColor, COLOR_HSV2BGR);
}

void trackbarTint(int pos, void*)
{
    Mat Color;
    BaseColor.copyTo(Color);
    Rect r(pos - 1, 0, 4, 20);
    rectangle(Color, r, Scalar(0));
    imshow(nameWinParam, Color);
    ColorTint = BaseColor.at<Vec3b>(0, pos);

    float den = (float)valueDen / 100.0f;
    tint(img, filterImg, ColorTint, den);
    imshow(nameWinFilter, filterImg);
}

void trackbarDen(int, void *)
{
    float den = (float)valueDen / 100.0f;
    tint(img, filterImg, ColorTint, den);
    imshow(nameWinFilter, filterImg);
}
