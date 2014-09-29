#include <opencv2/photoeffects.hpp>
#include <iostream>

using namespace cv;
using namespace std;

const char *ORIGINAL_IMAGE="Original image";
const char *FILM_GRAIN_IMAGE="Film Grain image";
const char *helper =
"./filmGrain_sample <img> <value of grain>\n\
\t<img> - file name contained the processed image\n\
\t<value of grain> - degree graininess\n\
";
int processArguments(int argc, char **argv, Mat &img, int &grainValue);

int main(int argc, char** argv)
{

    Mat src;
    int grainValue;
    if (processArguments(argc, argv, src, grainValue) != 0)
    {
        cout << helper << endl;
        return 1;
    }
    namedWindow(ORIGINAL_IMAGE, CV_WINDOW_AUTOSIZE);
    imshow(ORIGINAL_IMAGE, src);
    Mat dst;
    RNG rng=RNG(0);
    filmGrain(src, dst, grainValue, rng);
    imshow(FILM_GRAIN_IMAGE, dst);
    cout << "Press any key to EXIT"<<endl;
    waitKey(0);
    return 0;
}

int processArguments(int argc, char **argv, Mat &img, int &grainValue)
{
    if (argc < 3)
    {
        return 1;
    }
    img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    grainValue = atoi(argv[2]);
    return 0;
}
