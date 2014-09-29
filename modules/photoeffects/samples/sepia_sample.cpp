#include <opencv2/photoeffects.hpp>
#include <iostream>

using namespace cv;
using namespace std;

const char *helper =
"./sepia_sample <img>\n\
\t<img> - file name contained the processed image\n\
";

int processArguments(int argc, char **argv, Mat &img);

int main(int argc, char **argv)
{
    const char *srcImgWinName = "Initial image",
               *dstImgWinName = "Processed image";
    Mat img, sepiaImg;

    if (processArguments(argc, argv, img) != 0)
    {
        cout << helper << endl;
        return 1;
    }
    int opRes = sepia(img, sepiaImg);
    if (opRes == 1)
    {
        cout << "Incorrect image type." << endl;
        return 2;
    }

    namedWindow(srcImgWinName);
    namedWindow(dstImgWinName);
    imshow(srcImgWinName, img);
    imshow(dstImgWinName, sepiaImg);
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
    img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    return 0;
}
