// This file is part of the OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

//
//[1] David S. Bolme et al. "Visual Object Tracking using Adaptive Correlation Filters"
//    http://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf
//

//
// credits:
// Kun-Hsin Chen: for initial c++ code
// Cracki: for the idea of only converting the used patch to gray
//

#include "precomp.hpp"

#include "opencv2/tracking/tracking_legacy.hpp"

namespace cv {
inline namespace tracking {
namespace impl {
namespace {

struct DummyModel : detail::tracking::TrackerModel
{
     virtual void modelUpdateImpl() CV_OVERRIDE {}
     virtual void modelEstimationImpl( const std::vector<Mat>& ) CV_OVERRIDE {}
};

const double eps=0.00001;      // for normalization
const double rate=0.2;         // learning rate
const double psrThreshold=5.7; // no detection, if PSR is smaller than this

}  // namespace

struct MosseImpl CV_FINAL : legacy::TrackerMOSSE
{
protected:

    Point2d center; //center of the bounding box
    Size size;      //size of the bounding box
    Mat hanWin;
    Mat G;          //goal
    Mat H, A, B;    //state

    //  Element-wise division of complex numbers in src1 and src2
    Mat divDFTs( const Mat &src1, const Mat &src2 ) const
    {
        Mat c1[2],c2[2],a1,a2,s1,s2,denom,re,im;

        // split into re and im per src
        cv::split(src1, c1);
        cv::split(src2, c2);

        // (Re2*Re2 + Im2*Im2) = denom
        //   denom is same for both channels
        cv::multiply(c2[0], c2[0], s1);
        cv::multiply(c2[1], c2[1], s2);
        cv::add(s1, s2, denom);

        // (Re1*Re2 + Im1*Im1)/(Re2*Re2 + Im2*Im2) = Re
        cv::multiply(c1[0], c2[0], a1);
        cv::multiply(c1[1], c2[1], a2);
        cv::divide(a1+a2, denom, re, 1.0 );

        // (Im1*Re2 - Re1*Im2)/(Re2*Re2 + Im2*Im2) = Im
        cv::multiply(c1[1], c2[0], a1);
        cv::multiply(c1[0], c2[1], a2);
        cv::divide(a1+a2, denom, im, -1.0);

        // Merge Re and Im back into a complex matrix
        Mat dst, chn[] = {re,im};
        cv::merge(chn, 2, dst);
        return dst;
    }


    void preProcess( Mat &window ) const
    {
        window.convertTo(window, CV_32F);
        log(window + 1.0f, window);

        //normalize
        Scalar mean,StdDev;
        meanStdDev(window, mean, StdDev);
        window = (window-mean[0]) / (StdDev[0]+eps);

        //Gaussain weighting
        window = window.mul(hanWin);
    }


    double correlate( const Mat &image_sub, Point &delta_xy ) const
    {
        Mat IMAGE_SUB, RESPONSE, response;
        // filter in dft space
        dft(image_sub, IMAGE_SUB, DFT_COMPLEX_OUTPUT);
        mulSpectrums(IMAGE_SUB, H, RESPONSE, 0, true );
        idft(RESPONSE, response, DFT_SCALE|DFT_REAL_OUTPUT);
        // update center position
        double maxVal; Point maxLoc;
        minMaxLoc(response, 0, &maxVal, 0, &maxLoc);
        delta_xy.x = maxLoc.x - int(response.size().width/2);
        delta_xy.y = maxLoc.y - int(response.size().height/2);
        // normalize response
        Scalar mean,std;
        meanStdDev(response, mean, std);
        return (maxVal-mean[0]) / (std[0]+eps); // PSR
    }


    Mat randWarp( const Mat& a ) const
    {
        cv::RNG rng(8031965);

        // random rotation
        double C=0.1;
        double ang = rng.uniform(-C,C);
        double c=cos(ang), s=sin(ang);
        // affine warp matrix
        Mat_<float> W(2,3);
        W << c + rng.uniform(-C,C), -s + rng.uniform(-C,C), 0,
             s + rng.uniform(-C,C),  c + rng.uniform(-C,C), 0;

        // random translation
        Mat_<float> center_warp(2, 1);
        center_warp << a.cols/2, a.rows/2;
        W.col(2) = center_warp - (W.colRange(0, 2))*center_warp;

        Mat warped;
        warpAffine(a, warped, W, a.size(), BORDER_REFLECT);
        return warped;
    }


    virtual bool initImpl( const Mat& image, const Rect2d& boundingBox ) CV_OVERRIDE
    {
        model = makePtr<DummyModel>();

        Mat img;
        if (image.channels() == 1)
            img = image;
        else
            cvtColor(image, img, COLOR_BGR2GRAY);

        int w = getOptimalDFTSize(int(boundingBox.width));
        int h = getOptimalDFTSize(int(boundingBox.height));

        //Get the center position
        int x1 = int(floor((2*boundingBox.x+boundingBox.width-w)/2));
        int y1 = int(floor((2*boundingBox.y+boundingBox.height-h)/2));
        center.x = x1 + (w)/2;
        center.y = y1 + (h)/2;
        size.width = w;
        size.height = h;

        Mat window;
        getRectSubPix(img, size, center, window);
        createHanningWindow(hanWin, size, CV_32F);

        // goal
        Mat g=Mat::zeros(size,CV_32F);
        g.at<float>(h/2, w/2) = 1;
        GaussianBlur(g, g, Size(-1,-1), 2.0);
        double maxVal;
        minMaxLoc(g, 0, &maxVal);
        g = g / maxVal;
        dft(g, G, DFT_COMPLEX_OUTPUT);

        // initial A,B and H
        A = Mat::zeros(G.size(), G.type());
        B = Mat::zeros(G.size(), G.type());
        for(int i=0; i<8; i++)
        {
            Mat window_warp = randWarp(window);
            preProcess(window_warp);

            Mat WINDOW_WARP, A_i, B_i;
            dft(window_warp, WINDOW_WARP, DFT_COMPLEX_OUTPUT);
            mulSpectrums(G          , WINDOW_WARP, A_i, 0, true);
            mulSpectrums(WINDOW_WARP, WINDOW_WARP, B_i, 0, true);
            A+=A_i;
            B+=B_i;
        }
        H = divDFTs(A,B);
        return true;
    }

    virtual bool updateImpl( const Mat& image, Rect2d& boundingBox ) CV_OVERRIDE
    {
        if (H.empty()) // not initialized
            return false;

        Mat image_sub;
        getRectSubPix(image, size, center, image_sub);

        if (image_sub.channels() != 1)
            cvtColor(image_sub, image_sub, COLOR_BGR2GRAY);
        preProcess(image_sub);

        Point delta_xy;
        double PSR = correlate(image_sub, delta_xy);
        if (PSR < psrThreshold)
            return false;

        //update location
        center.x += delta_xy.x;
        center.y += delta_xy.y;

        Mat img_sub_new;
        getRectSubPix(image, size, center, img_sub_new);
        if (img_sub_new.channels() != 1)
            cvtColor(img_sub_new, img_sub_new, COLOR_BGR2GRAY);
        preProcess(img_sub_new);

        // new state for A and B
        Mat F, A_new, B_new;
        dft(img_sub_new, F, DFT_COMPLEX_OUTPUT);
        mulSpectrums(G, F, A_new, 0, true );
        mulSpectrums(F, F, B_new, 0, true );

        // update A ,B, and H
        A = A*(1-rate) + A_new*rate;
        B = B*(1-rate) + B_new*rate;
        H = divDFTs(A, B);

        // return tracked rect
        double x=center.x, y=center.y;
        int w = size.width, h=size.height;
        boundingBox = Rect2d(Point2d(x-0.5*w, y-0.5*h), Point2d(x+0.5*w, y+0.5*h));
        return true;
    }

public:
    MosseImpl() { isInit = 0; }

    // dummy implementation.
    virtual void read( const FileNode& ) CV_OVERRIDE {}
    virtual void write( FileStorage& ) const CV_OVERRIDE {}

}; // MosseImpl

}}  // namespace

Ptr<legacy::tracking::TrackerMOSSE> legacy::tracking::TrackerMOSSE::create()
{
    return makePtr<impl::MosseImpl>();
}


}  // namespace
