// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*******************************************************************************\
*                   Ridge Detection Filter                                     *
* This code implements ridge detection based on the Hessian Matrix             *
* Author: Venkatesh Vijaykumar GATech/ 2017                                    *
********************************************************************************/

#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc/ridgefilter.hpp"

namespace cv
{
namespace ximgproc
{
class RidgeDetectionFilterImpl : public RidgeDetectionFilter
{
    public:
        RidgeDetectionFilterImpl() {}

        virtual void getRidges(InputArray img, OutputArray out);

    private:
        virtual void getSobelX(InputArray img, OutputArray out);
        virtual void getSobelY(InputArray img, OutputArray out);
};

void RidgeDetectionFilterImpl::getSobelX(InputArray _img, OutputArray _out)
{
    _out.create(_img.size(), CV_32F);
    Sobel(_img, _out, CV_32F,1,0,3);
}

void RidgeDetectionFilterImpl::getSobelY(InputArray _img, OutputArray _out)
{
    _out.create(_img.size(), CV_32F);
    Sobel(_img, _out, CV_32F,0,1,3);
}

void RidgeDetectionFilterImpl::getRidges(InputArray _img, OutputArray _out)
{
    Mat img = _img.getMat();
    CV_Assert(!img.empty());
    CV_Assert(img.channels() == 1 || img.channels() == 3);

    if(img.channels() == 3)
        cvtColor(img, img, COLOR_BGR2GRAY);

    Mat sbx;
    getSobelX(img, sbx);
    Mat sby;
    getSobelY(img, sby);
    Mat sbxx;
    getSobelX(sbx, sbxx);
    Mat sbyy;
    getSobelY(sby, sbyy);
    Mat sbxy;
    getSobelY(sbx, sbxy);
    Mat sb2xx;
    multiply(sbxx,sbxx,sb2xx);
    Mat sb2yy;
    multiply(sbyy,sbyy,sb2yy);
    Mat sb2xy;
    multiply(sbxy,sbxy,sb2xy);
    Mat sbxxyy;
    multiply(sbxx,sbyy,sbxxyy);
    Mat rootex;
    rootex = (sb2xx + (sb2xy + sb2xy + sb2xy + sb2xy) - ((sbxxyy) + (sbxxyy)) +sb2yy);
    Mat root;
    sqrt(rootex, root);
    Mat ridgexp;
    ridgexp = ((sbxx + sbyy) + (root));
    Mat ridges;
    ridges = ((ridgexp) / 2);
    ridges.copyTo(_out);
}

Ptr<RidgeDetectionFilter> RidgeDetectionFilter::create()
{
    return makePtr<RidgeDetectionFilterImpl>();
}

}
}
