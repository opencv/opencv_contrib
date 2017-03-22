#include "precomp.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc/ridgefilter.hpp"
// got rid of dual iostream includes
#include <iostream>

namespace cv{
    namespace ximgproc{
        namespace ridgefilter{

            class RidgeDetectionFilterImpl : public RidgeDetectionFilter {
                public:
                    RidgeDetectionFilterImpl() {
                        name_ = "RidgeDetectionFilter";
                    }

                    virtual void getRidges(InputArray img, OutputArray out);

                private:
                    String name_;
                    virtual void getSobelX(InputArray img, OutputArray out);
                    virtual void getSobelY(InputArray img, OutputArray out);
            };

            void RidgeDetectionFilterImpl::getSobelX(InputArray _img, OutputArray _out) {
                Mat img = _img.getMat();
                _out.create(img.size(), CV_32F);
                Mat out = _out.getMat();
                cvtColor(img, img, COLOR_BGR2GRAY);
                Sobel(img, out, CV_32F,1,0,3);
            }

            void RidgeDetectionFilterImpl::getSobelY(InputArray _img, OutputArray _out) {
                Mat img = _img.getMat();
                _out.create(img.size(), CV_32F);
                Mat out = _out.getMat();
                cvtColor(img, img, COLOR_BGR2GRAY);
                Sobel(img, out, CV_32F,0,1,3);
            }

            void RidgeDetectionFilterImpl::getRidges(InputArray _img, OutputArray _out){
                Mat img = _img.getMat();
                _out.create(img.size(), CV_32F);
                Mat out = _out.getMat();
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
                out = ridges;
                //_out = out;
            }
            Ptr<RidgeDetectionFilter> createRidgeDetectionFilter() {
                Ptr<RidgeDetectionFilter> s = makePtr<RidgeDetectionFilterImpl>();
                return s;
            }
        }
    }
}