#include <opencv2/photoeffects.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <iostream>

using namespace cv;
using namespace cv::photoeffects;
using namespace std;

const char *helper =
"./warmify_sample <img>\n\
\t<img> - file name contained the processed image\n\
";

int processArguments(int argc, char **argv, Mat &img);

int main(int argc, char** argv)
{
    const char *srcImgWinName = "Initial image",
               *dstImgWinName = "Processed image";
    Mat image, warmifyImg;

    if (processArguments(argc, argv, image))
    {
        cout << helper << endl;
        return 1;
    }

    try
    {
        warmify(image, warmifyImg, 30);
    }
    catch(...)
    {
        cout << "Incorrect image type." << endl;
        return 2;
    }

    namedWindow(srcImgWinName);
    namedWindow(dstImgWinName);
    imshow(srcImgWinName, image);
    imshow(dstImgWinName, warmifyImg);
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
