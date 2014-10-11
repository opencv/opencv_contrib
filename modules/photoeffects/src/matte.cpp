#include "precomp.hpp"


using namespace std;
namespace cv { namespace photoeffects {


void matte(InputArray src, OutputArray dst, float sigma=25)
{
    CV_Assert((src.type() == CV_8UC3) || (src.type() == CV_32FC3));
    CV_Assert((sigma > 0.0f));

    Mat imgSrc = src.getMat();

    dst.create(src.size(), CV_8UC3);
    Mat imgDst = dst.getMat();

    imgSrc.copyTo(imgDst);


    float centerX , centerY;
    centerX = imgDst.cols/2.0f;
    centerY = imgDst.rows/2.0f;

    Size s;

    s.height = imgDst.rows/sigma*10;
    s.width = imgDst.cols/sigma*10;

    float aSquare = s.height * s.height / 4.0f;
    float bSquare = s.width * s.width / 4.0f;

    float radiusMax = centerY * centerY/ aSquare + centerX * centerX / bSquare - 1.0f;


    for (int i =0;i<imgDst.rows;i++)
    {
        for (int j=0;j<imgDst.cols;j++)
        {

            float dist= (i-centerY)*(i-centerY)/aSquare + (j-centerX)*(j-centerX)/bSquare;
            float coef = 1.0f;
            if (dist > 1.0f)
            {
                coef = 1.0f -(dist - 1.0f) / radiusMax;
                coef = -2*coef*coef*coef*coef*coef + 4*coef*coef*coef - coef;
            }

            imgDst.at<Vec3b>(i, j) = imgDst.at<Vec3b>(i, j)*coef+
                    Vec3b(255, 255, 255)*(1.0f-coef);

        }
    }
}
}}
