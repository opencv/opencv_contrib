#include <opencv2/photoeffects.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <iostream>

using namespace cv;
using namespace cv::photoeffects;
using namespace std;

const char *helper =
"./edgeBlur_sample <img>\n\
\t<img> - file name contained the processed image\n\
";
const char *nameWinImage = "Image";
const char *nameWinFilter = "Edge blur";

Mat img, filterImg;
int indentTop = 0, indentLeft = 0;

void trackbarIndTop(int pos, void *);
void trackbarIndLeft(int pos, void *);
int processArguments(int argc, char** argv, Mat &image);

int main(int argc, char** argv)
{
    if (processArguments(argc, argv, img) != 0)
    {
        cout << helper << endl;
        return 1;
    }

    namedWindow(nameWinFilter);
    createTrackbar("Indent top", nameWinFilter, &indentTop,
                    img.rows / 2 - 10, trackbarIndTop);
    createTrackbar("Indent left", nameWinFilter, &indentLeft,
                    img.cols / 2 - 10, trackbarIndLeft);

    namedWindow(nameWinImage);

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

void trackbarIndTop(int pos, void*)
{
    edgeBlur(img, filterImg, indentTop, indentLeft);
    imshow(nameWinFilter, filterImg);
}

void trackbarIndLeft(int pos, void*)
{
    edgeBlur(img, filterImg, indentTop, indentLeft);
    imshow(nameWinFilter, filterImg);
}
