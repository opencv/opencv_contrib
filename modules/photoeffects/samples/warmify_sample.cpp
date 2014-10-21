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
const char *srcImgWinName = "Initial image",
           *dstImgWinName = "Processed image";

int warmRatio;
Mat image, warmifyImg;

int processArguments(int argc, char **argv, Mat &img);
void on_trackbar(int, void*);

int main(int argc, char** argv)
{
    if (processArguments(argc, argv, image))
    {
        cout << helper << endl;
        return 1;
    }

    namedWindow(srcImgWinName, WINDOW_FREERATIO);
    namedWindow(dstImgWinName, WINDOW_FREERATIO);

    warmRatio = 0;
    createTrackbar("Warm Ratio", srcImgWinName, &warmRatio, 255, on_trackbar);

    imshow(srcImgWinName, image);
    imshow(dstImgWinName, image);
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

void on_trackbar(int, void*)
{
    warmify(image, warmifyImg, warmRatio);
    imshow(dstImgWinName, warmifyImg);
}
