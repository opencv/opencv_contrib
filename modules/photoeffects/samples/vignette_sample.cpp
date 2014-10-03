#include <opencv2/photoeffects.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <iostream>

using namespace cv;
using namespace cv::photoeffects;
using namespace std;

const char *helper =
"./vignette_sample <img>\n\
\t<img> - file name contained the processed image\n\
";

int processArguments(int argc, char **argv, Mat &img);

int main(int argc, char** argv)
{
    const char *srcImgWinName = "Initial image",
               *dstImgWinName = "Processed image";
    Mat image, vignetteImg;
    Size rectangle;

    if (processArguments(argc, argv, image) != 0)
    {
        cout << helper << endl;
        return 1;
    }

    rectangle.height = image.rows / 1.5f;
    rectangle.width = image.cols / 2.0f;

    try
    {
        vignette(image, vignetteImg, rectangle);
    }
    catch(...)
    {
        cout << "Incorrect image type, size of rectangle or image wasn't found." << endl;
        return 2;
    }

    namedWindow(srcImgWinName);
    namedWindow(dstImgWinName);
    imshow(srcImgWinName, image);
    imshow(dstImgWinName, vignetteImg);
    waitKey();
    destroyAllWindows();
    return 0;
}

int processArguments(int argc, char **argv, Mat &img)
{
    if (argc < 2)
    {
        return 1;
    }
    img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    return 0;
}
