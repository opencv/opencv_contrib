#include <opencv2/photoeffects.hpp>
#include <iostream>

using namespace cv;
using namespace std;

const char *ORIGINAL_IMAGE="Original image";
const char *FADED_IMAGE="Faded image";
const char *helper =
"./fadeColor_sample <img>\n\
\t<img> - file name contained the processed image\n\
";

Point startPoint,endPoint;
int numberChoosenPoint=0;

void CallBackFunc(int event, int x, int y, int flags, void* userdata);
int processArguments(int argc, char **argv, Mat &img);

int main(int argc, char** argv)
{
    Mat src;
    if (processArguments(argc, argv, src) != 0)
    {
        cout << helper << endl;
        return 1;
    }
    namedWindow(ORIGINAL_IMAGE, CV_WINDOW_AUTOSIZE);
    imshow(ORIGINAL_IMAGE, src);
    setMouseCallback(ORIGINAL_IMAGE, CallBackFunc, &src);
    cout << "Choose two points on image and press any key."<<endl;
    waitKey(0);
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
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    Mat src=*((Mat*)userdata);
    Mat srcCopy;
    src.copyTo(srcCopy);
    if  ( event == EVENT_LBUTTONDOWN )
    {
        switch(numberChoosenPoint)
        {
        case 0:
            numberChoosenPoint++;
            startPoint=Point(x, y);
            cout<<"x:"<<startPoint.x<<endl<<"y:"<<startPoint.y<<endl;
            circle(srcCopy, Point(x,y), 5, CV_RGB(255,50,255) ,4);
            imshow(ORIGINAL_IMAGE, srcCopy);
            break;
        case 1:
            numberChoosenPoint++;
            endPoint=Point(x, y);
            cout<<"x:"<<endPoint.x<<endl<<"y:"<<endPoint.y<<endl;
            circle(srcCopy, startPoint, 5, CV_RGB(255,50,255), 4);
            circle(srcCopy, endPoint, 5, CV_RGB(255,50,255), 4);
            Mat dst;
            fadeColor(src, dst, startPoint, endPoint);
            imshow(ORIGINAL_IMAGE, srcCopy);
            namedWindow(FADED_IMAGE, CV_WINDOW_AUTOSIZE);
            imshow(FADED_IMAGE, dst);
            break;
        }
    }
}
