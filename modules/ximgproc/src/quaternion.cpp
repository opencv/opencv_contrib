// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.#include "precomp.hpp"
#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc/color_match.hpp"

using namespace std;

namespace cv { namespace ximgproc {

void createQuaternionImage(InputArray _img, OutputArray _qimg)
{
    int type = _img.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_CheckType(depth, depth == CV_8U || depth == CV_32F || depth == CV_64F, "Depth must be CV_8U, CV_32F or CV_64F");
    CV_Assert(_img.dims() == 2 && cn == 3);
    vector<Mat> qplane(4);
    vector<Mat> plane;
    split(_img, plane);
    qplane[0] = Mat::zeros(_img.size(), CV_64FC1);
    for (int i = 0; i < cn; i++)
        plane[i].convertTo(qplane[3-i], CV_64F);
    merge(qplane, _qimg);
}

void qconj(InputArray _img, OutputArray _qimg)
{
    int type = _img.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_CheckType(depth, depth == CV_32F || depth == CV_64F, "Depth must be CV_32F or CV_64F");
    CV_Assert(_img.dims() == 2 && cn == 4);
    vector<Mat> qplane(4), plane;
    split(_img, plane);
    qplane[0] = plane[0];
    qplane[1] = -plane[1];
    qplane[2] = -plane[2];
    qplane[3] = -plane[3];
    merge(qplane, _qimg);
}

void qunitary(InputArray _img, OutputArray _qimg)
{
    int type = _img.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert((depth == CV_64F) && _img.dims() == 2 && cn == 4);
    _img.copyTo(_qimg);
    Mat qimg = _qimg.getMat();
    qimg.forEach<Vec4d>([](Vec4d &p, const int * /*position*/) -> void {
        double d = p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + p[3] * p[3];
        d = 1 / sqrt(d);
        p *= d;
    });
}

void qdft(InputArray _img, OutputArray _qimg, int  	flags, bool sideLeft)
{
    CV_INSTRUMENT_REGION();

    int type = _img.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert(depth == CV_64F && _img.dims() == 2 && cn == 4);
    float c;
    if (sideLeft)
        c = 1;  // Left qdft
    else
        c = -1; // right qdft

    vector<Mat> q;
    Mat img;
    img = _img.getMat();

    CV_Assert(getOptimalDFTSize(img.rows) == img.rows && getOptimalDFTSize(img.cols) == img.cols);

    split(img, q);
    Mat c1r;
    Mat c1i; // Imaginary part of c1 =x'
    Mat c2r; // Real part of c2 =y'
    Mat c2i; // Imaginary part of c2=z'
    c1r = q[0].clone();
    c1i = (q[1] + q[2] + q[3]) / sqrt(3);
    c2r = (q[2] - q[3]) / sqrt(2);
    c2i = c * (q[3] + q[2] - 2 * q[1]) / sqrt(6);
    vector<Mat> vc1 = { c1r,c1i }, vc2 = { c2r,c2i };
    Mat c1, c2, C1, C2;
    merge(vc1, c1);
    merge(vc2, c2);
    if (flags& DFT_INVERSE)
    {
        dft(c1, C1, DFT_COMPLEX_OUTPUT | DFT_INVERSE|DFT_SCALE);
        dft(c2, C2, DFT_COMPLEX_OUTPUT | DFT_INVERSE | DFT_SCALE);
    }
    else
    {
        dft(c1, C1, DFT_COMPLEX_OUTPUT);
        dft(c2, C2, DFT_COMPLEX_OUTPUT);
    }
    split(C1, vc1);
    split(C2, vc2);
    vector<Mat> qdft(4);
    qdft[0] = vc1[0].clone();
    qdft[1] = vc1[1] / sqrt(3) - c * 2 * vc2[1] / sqrt(6);
    qdft[2] = vc1[1] / sqrt(3) + vc2[0] / sqrt(2) + c * vc2[1] / sqrt(6);
    qdft[3] = vc1[1] / sqrt(3) - vc2[0] / sqrt(2) + c * vc2[1] / sqrt(6);
    Mat dst0;
    merge(qdft, dst0);
    dst0.copyTo(_qimg);
}


void qmultiply(InputArray  	src1, InputArray  	src2, OutputArray  	dst)
{
    int type = src1.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert(depth == CV_64F && src1.dims() == 2 && cn == 4);
    type = src2.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert(depth == CV_64F && src2.dims() == 2 && cn == 4);
    vector<Mat> q3(4);
    if (src1.rows() == src2.rows() && src1.cols() == src2.cols())
    {
        vector<Mat> q1, q2;
        split(src1, q1);
        split(src2, q2);
        q3[0] = q1[0].mul(q2[0]) - q1[1].mul(q2[1]) - q1[2].mul(q2[2]) - q1[3].mul(q2[3]);
        q3[1] = q1[0].mul(q2[1]) + q1[1].mul(q2[0]) + q1[2].mul(q2[3]) - q1[3].mul(q2[2]);
        q3[2] = q1[0].mul(q2[2]) - q1[1].mul(q2[3]) + q1[2].mul(q2[0]) + q1[3].mul(q2[1]);
        q3[3] = q1[0].mul(q2[3]) + q1[1].mul(q2[2]) - q1[2].mul(q2[1]) + q1[3].mul(q2[0]);
    }
    else if (src1.rows() == 1 && src1.cols() == 1)
    {
        vector<Mat> q2;
        Vec4d q1 = src1.getMat().at<Vec4d>(0, 0);
        split(src2, q2);
        q3[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
        q3[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
        q3[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
        q3[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
    }
    else if (src2.rows() == 1 && src2.cols() == 1)
    {
        vector<Mat> q1;
        split(src1, q1);
        Vec4d q2 = src2.getMat().at<Vec4d>(0, 0);
        q3[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
        q3[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
        q3[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
        q3[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
    }
    else
        CV_Assert(src1.rows() == src2.rows() && src1.cols() == src2.cols());
    merge(q3, dst);

}

void colorMatchTemplate(InputArray _image, InputArray _templ, OutputArray _result)
{
    CV_INSTRUMENT_REGION();
    Mat image = _image.getMat(), imageF;
    CV_Assert(image.channels() == 3);
    Mat colorTemplate = _templ.getMat(), colorTemplateF;
    CV_Assert(colorTemplate.channels() == 3);
    int rr = max(getOptimalDFTSize(image.rows), getOptimalDFTSize(colorTemplate.rows));
    int cc = max(getOptimalDFTSize(image.cols), getOptimalDFTSize(colorTemplate.cols));
    Mat logo(rr, cc, CV_64FC3, Scalar::all(0));
    Mat img = Mat(rr, cc, CV_64FC3, Scalar::all(0));
    colorTemplate.convertTo(colorTemplateF, CV_64F, 1 / 256.),
    colorTemplateF.copyTo(logo(Rect(0, 0, colorTemplate.cols, colorTemplate.rows)));
    image.convertTo(imageF, CV_64F, 1 / 256.);
    imageF.copyTo(img(Rect(0, 0, image.cols, image.rows)));
    Mat qimg, qlogo;
    Mat qimgFFT, qimgIFFT, qlogoFFT;
    // Create quaternion image
    createQuaternionImage(img, qimg);
    createQuaternionImage(logo, qlogo);
    // quaternion fourier transform
    qdft(qimg, qimgFFT, 0, true);
    qdft(qimg, qimgIFFT, DFT_INVERSE, true);
    qdft(qlogo, qlogoFFT, 0, false);
    double sqrtnn = sqrt(static_cast<int>(qimgFFT.rows*qimgFFT.cols));
    qimgFFT /= sqrtnn;
    qimgIFFT *= sqrtnn;
    qlogoFFT /= sqrtnn;
    Mat mu(1, 1, CV_64FC4, Scalar(0, 1, 1, 1) / sqrt(3.));
    Mat qtmp, qlogopara, qlogoortho;
    qmultiply(mu, qlogoFFT, qtmp);
    qmultiply(qtmp, mu, qtmp);
    subtract(qlogoFFT, qtmp, qlogopara);
    qlogopara = qlogopara / 2;
    subtract(qlogoFFT, qlogopara, qlogoortho);
    Mat qcross1, qcross2, cqf, cqfi;
    qconj(qimgFFT, cqf);
    qconj(qimgIFFT, cqfi);
    qmultiply(cqf, qlogopara, qcross1);
    qmultiply(cqfi, qlogoortho, qcross2);
    Mat pwsp = qcross1 + qcross2;
    Mat crossCorr, pwspUnitary;
    qunitary(pwsp, pwspUnitary);
    qdft(pwspUnitary, crossCorr, DFT_INVERSE, false);
    vector<Mat> p;
    split(crossCorr, p);
    Mat imgcorr = (p[0].mul(p[0]) + p[1].mul(p[1]) + p[2].mul(p[2]) + p[3].mul(p[3]));
    sqrt(imgcorr, _result);
}
}
}
