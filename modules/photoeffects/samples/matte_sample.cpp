#include <opencv2/photoeffects.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <iostream>

using namespace cv;

using namespace cv::photoeffects;
using namespace std;

const string ORIGINAL_IMAGE = "Original image";
const string MATTE_IMAGE = "Matte image";
const char *helper = "./matte_sample <img> <sigma1> <sigma2>\n\
\t<img>-file name contained the processed image\n\
\t<sigma>-float param - power of the blur";


int processArguments(int argc, char **argv, Mat &src,float &sigma);
int main(int argc, char** argv)
{
    float sigma;
    Mat src, matteImage;
    if ( processArguments ( argc , argv , src,sigma) != 0)
    {
        cout << helper << endl;
        return 1;
    }
    namedWindow ( ORIGINAL_IMAGE , WINDOW_AUTOSIZE ) ;
    imshow ( ORIGINAL_IMAGE , src );
    matte ( src , matteImage , sigma );
    namedWindow ( MATTE_IMAGE , WINDOW_AUTOSIZE);
    imshow ( MATTE_IMAGE , matteImage);
    waitKey(0);
    destroyAllWindows();
    return 0;
}

int processArguments(int argc, char **argv, Mat &src, float &sigma)
{
    if ( argc < 3 )
    {
        return 1;
    }
    src = imread ( argv[1] , CV_LOAD_IMAGE_COLOR) ;
    sigma = atof ( argv[2] );

    return 0;
}
