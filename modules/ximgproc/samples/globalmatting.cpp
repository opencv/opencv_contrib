#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace ximgproc;

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        cout << "arg1: location of input image" << endl;
        cout << "arg2: location of its trimap" << endl;
        cout << "arg3(optional): number of iterations to run expansion of trimap" << endl;
        return -1;
    }
    string img_path = argv[1];
    string tri_path = argv[2];
    int niter = 9;
    if (argc == 4)
    {
        niter = atoi(argv[3]);
    }
    Mat image = imread(img_path, IMREAD_COLOR);
    Mat trimap = imread(tri_path, IMREAD_GRAYSCALE);
    if (image.empty() || trimap.empty())
    {
        cout << "Could not load the inputs" << endl;
        return -2;
    }
    // (optional) exploit the affinity of neighboring pixels to reduce the
    // size of the unknown region. please refer to the paper
    // 'Shared Sampling for Real-Time Alpha Matting'.

    Mat foreground, alpha;

    Ptr<GlobalMatting> gm = createGlobalMatting();

    gm->getMat(image, trimap, foreground, alpha, niter);

    imwrite("alpha-matte.png", alpha);

    imshow("input", image);
    imshow("trimap", trimap);
    imshow("alpha-matte", alpha);
    waitKey();

    return 0;
}
