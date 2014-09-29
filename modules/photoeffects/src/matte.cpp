#include "precomp.hpp"

namespace cv { namespace photoeffects {

namespace
{
    Point topleftFind(Point firstPoint, Point secondPoint, int &xsize, int &ysize)
    {
        Point topleft(0, 0);
        if(xsize < 0)
        {
            topleft.x = firstPoint.x;
            xsize = -xsize;
        }
        else
        {
            topleft.x = secondPoint.x;
        }
        if(ysize < 0)
        {
            topleft.y = secondPoint.y;
            ysize = -ysize;
        }
        else
        {
            topleft.y = firstPoint.y;
        }
        return topleft;
    }
}

int matte(InputArray src, OutputArray dst, Point firstPoint, Point secondPoint, float sigmaX,
            float sigmaY, Size ksize)
{
    CV_Assert((src.type() == CV_8UC3) || (src.type() == CV_32FC3));
    CV_Assert((sigmaX > 0.0f) || (sigmaY > 0.0f));
    Mat srcImg = src.getMat();
    CV_Assert(!(srcImg.empty()));
    if(srcImg.type() != CV_32FC3)
    {
        srcImg.convertTo(srcImg, CV_32FC3, 1.0f/255.0f);
    }
    int xsize = firstPoint.x - secondPoint.x;
    int ysize = firstPoint.y - secondPoint.y;
    Point topLeft = topleftFind(firstPoint, secondPoint, xsize, ysize);
    const Scalar black = Scalar(0.0f,0.0f,0.0f);
    const Scalar white = Scalar(1.0f,1.0f,1.0f);
    Mat mask(srcImg.rows, srcImg.cols, CV_32FC1, black);
    ellipse(mask, Point((topLeft.x+xsize/2),(topLeft.y-ysize/2)),
            Size(xsize/2,ysize/2), 0, 0, 360, white, -1);
    GaussianBlur(mask, mask, ksize, sigmaX, sigmaY);
    vector<Mat> ch_img;
    split(srcImg,ch_img);
    ch_img[0]=ch_img[0].mul(mask)+1.0f-mask;
    ch_img[1]=ch_img[1].mul(mask)+1.0f-mask;
    ch_img[2]=ch_img[2].mul(mask)+1.0f-mask;
    merge(ch_img,dst);
    return 0;
}

}}