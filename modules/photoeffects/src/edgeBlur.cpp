#include "precomp.hpp"

namespace cv { namespace photoeffects {

int edgeBlur(InputArray src, OutputArray dst, int indentTop, int indentLeft)
{
    CV_Assert(!src.empty());
    CV_Assert(src.type() == CV_8UC3);
    dst.create(src.size(), src.type());
    Mat image = src.getMat(), outputImage = dst.getMat();

    CV_Assert(indentTop >= 0 && indentTop <= (image.rows / 2 - 10));
    CV_Assert(indentLeft >= 0 && indentLeft <= (image.cols / 2 - 10));

    float halfWidth = image.cols / 2.0f;
    float halfHeight = image.rows / 2.0f;
    float a = (halfWidth - indentLeft)
            * (halfWidth - indentLeft);
    float b = (halfHeight - indentTop)
            * (halfHeight - indentTop);
    int kSizeEdges = halfWidth * halfWidth / a + halfHeight * halfHeight / b;

    // 15 is a maximal kernel size
    kSizeEdges = MIN(kSizeEdges, 15);
    Mat bearingImage;
    copyMakeBorder(image, bearingImage, kSizeEdges, kSizeEdges,
        kSizeEdges, kSizeEdges, BORDER_REPLICATE);


    for (int i = kSizeEdges; i < bearingImage.rows - kSizeEdges; i++)
    {
        for (int j = kSizeEdges; j < bearingImage.cols - kSizeEdges; j++)
        {
            float radius = (halfHeight - i)
                    * (halfHeight- i)
                    / b
                    + (halfWidth - j)
                    * (halfWidth - j)
                    / a;
            if (radius < 1.0f)
            {
                outputImage.at<Vec3b>(i - kSizeEdges, j - kSizeEdges) =
                    bearingImage.at<Vec3b>(i, j);
                continue;
            }
            int size = MIN(radius, kSizeEdges);
            radius = 2.0f * (radius - 0.5f) * (radius - 0.5f);
            float sumC = 0.0f;
            Vec3f sumF;
            float coeff1 = 1.0f / (CV_PI * radius);
            for (int x = -size; x <= size; x++)
            {
                for (int y = -size; y <= size; y++)
                {
                    float coeff2 = coeff1 * exp(- (x * x + y * y) / radius);
                    Vec3b Color = bearingImage.at<Vec3b>(x + i, y + j);
                    sumF += coeff2 * (Vec3f)Color;
                    sumC += coeff2;
                }
            }
            sumF *= (1.0f / sumC);
            outputImage.at<Vec3b>(i - kSizeEdges,
                                  j - kSizeEdges) = (Vec3b)sumF;
        }
    }

    return 0;
}

}}
