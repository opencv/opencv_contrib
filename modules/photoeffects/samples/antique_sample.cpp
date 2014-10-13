#include <opencv2/photoeffects.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <iostream>

using namespace cv;
using namespace cv::photoeffects;
using namespace std;

const string ORIGINAL_IMAGE = "Original image";
const string ANTIQUE_IMAGE = "Antique image";

const char* helper = "./antique_sample <src> <texture> <alpha>\n\
\t<src> - file name contained the source image, must be 3-channel RGB-image\n\
\t<texture> - file name contained the texture image, must be 3-channel RGB-image\n\
\t<alpha> - float coefficient of intensity texture applying, must be real between 0 to 1\n";

int processArguments(int argc, char** argv, Mat &src, Mat &texture, float &alpha);

int main(int argc, char** argv)
{
    Mat src;
    Mat texture;
    float alpha;
    Mat dst;
    if(processArguments(argc, argv, src, texture, alpha) != 0)
    {
        cout<< helper << endl;
        return 1;
    }
    namedWindow(ORIGINAL_IMAGE, WINDOW_AUTOSIZE);
    imshow(ORIGINAL_IMAGE,src);
    antique(src, dst, texture, alpha);
    namedWindow(ANTIQUE_IMAGE, WINDOW_AUTOSIZE);
    imshow(ANTIQUE_IMAGE,dst);
    cout<<"Press any key"<<endl;
    waitKey(0);
    destroyAllWindows();
    return 0;
}

int processArguments(int argc, char** argv, Mat &src, Mat &texture, float &alpha)
{
    if(argc < 4)
    {
        return 1;
    }
    src = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    texture = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    alpha = atof(argv[3]);
    return 0;
}