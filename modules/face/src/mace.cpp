#include "precomp.hpp"
#include "opencv2/face/mace.hpp"

namespace cv {
namespace face {


//
//! Rearrange the quadrants of Fourier image
//!  so that the origin is at the image center
//
static void shiftDFT(const Mat &src, Mat &dst)
{
    Size size = src.size();

    if (dst.empty() || (dst.size().width != size.width || dst.size().height != size.height))
    {
        dst.create(src.size(), src.type());
    }

    int cx = size.width/2;
    int cy = size.height/2; // image center

    Mat q1 = src(Rect(0, 0, cx,cy));
    Mat q2 = src(Rect(cx,0, cx,cy));
    Mat q3 = src(Rect(cx,cy,cx,cy));
    Mat q4 = src(Rect(0, cy,cx,cy));
    Mat d1 = dst(Rect(0, 0, cx,cy));
    Mat d2 = dst(Rect(cx,0, cx,cy));
    Mat d3 = dst(Rect(cx,cy,cx,cy));
    Mat d4 = dst(Rect(0, cy,cx,cy));

    if (src.data != dst.data){
        q3.copyTo(d1);
        q4.copyTo(d2);
        q1.copyTo(d3);
        q2.copyTo(d4);
    } else {
        Mat tmp;
        q3.copyTo(tmp);
        q1.copyTo(d3);
        tmp.copyTo(d1);
        q4.copyTo(tmp);
        q2.copyTo(d4);
        tmp.copyTo(d2);
    }
}


// Computes 64-bit "cyclic redundancy check" sum, as specified in ECMA-182
static uint64 crc64( const uchar* data, size_t size, uint64 crc0=0 )
{
    static uint64 table[256];
    static bool initialized = false;

    if( !initialized )
    {
        for( int i = 0; i < 256; i++ )
        {
            uint64 c = i;
            for( int j = 0; j < 8; j++ )
                c = ((c & 1) ? CV_BIG_UINT(0xc96c5795d7870f42) : 0) ^ (c >> 1);
            table[i] = c;
        }
        initialized = true;
    }

    uint64 crc = ~crc0;
    for( size_t idx = 0; idx < size; idx++ )
        crc = table[(uchar)crc ^ data[idx]] ^ (crc >> 8);

    return ~crc;
}

struct MACEImpl CV_FINAL : MACE {
    Mat_<Vec2d> maceFilter; // filled from compute()
    Mat convFilter;         // optional random convolution (cancellable)
    int IMGSIZE;            // images will get resized to this
    double threshold;       // minimal "sameness" threshold from the train images


    MACEImpl(int siz) : IMGSIZE(siz), threshold(DBL_MAX) {}

    void salt(const String &passphrase) CV_OVERRIDE {
        theRNG().state = ((int64)crc64((uchar*)passphrase.c_str(), passphrase.size()));
        convFilter.create(IMGSIZE, IMGSIZE, CV_64F);
        randn(convFilter, 0, 1.0/(IMGSIZE*IMGSIZE));
    }


    Mat dftImage(Mat img) const {
        Mat gray;
        resize(img, gray, Size(IMGSIZE,IMGSIZE)) ;
        if (gray.channels() > 1)
            cvtColor(gray, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);
        gray.convertTo(gray, CV_64F);
        if (! convFilter.empty()) { // optional, but unfortunately, it has to happen after resize/equalize ops.
            filter2D(gray, gray, CV_64F, convFilter);
        }
        Mat input[2] = {gray, Mat(gray.size(), gray.type(), 0.0)};
        Mat complexInput;
        merge(input, 2, complexInput);

        Mat_<Vec2d> dftImg(IMGSIZE*2, IMGSIZE*2, 0.0);
        complexInput.copyTo(dftImg(Rect(0,0,IMGSIZE,IMGSIZE)));

        dft(dftImg, dftImg);
        return dftImg;
    }


    // compute the mace filter: `h = D(-1) * X * (X(+) * D(-1) * X)(-1) * C`
    void compute(std::vector<Mat> images) {
        return compute(images, false);
    }
    void compute(std::vector<Mat> images, bool isdft) {
        int size = (int)images.size();
        int IMGSIZE_2X = IMGSIZE * 2;
        int TOTALPIXEL = IMGSIZE_2X * IMGSIZE_2X;

        Mat_<Vec2d> D(TOTALPIXEL, 1, 0.0);
        Mat_<Vec2d> S(TOTALPIXEL, size, 0.0);
        Mat_<Vec2d> SPLUS(size, TOTALPIXEL, 0.0);
        for (int i=0; i<size; i++) {
            Mat_<Vec2d> dftImg = isdft ? images[i] : dftImage(images[i]);
            for (int l=0; l<IMGSIZE_2X; l++) {
                for (int m=0; m<IMGSIZE_2X; m++) {
                    int j = l * IMGSIZE_2X + m;
                    Vec2d s = dftImg(l, m);
                    S(j, i) = s;
                    SPLUS(i, j) = Vec2d(s[0], -s[1]);
                    D(j, 0)[0] += (s[0]*s[0]) + (s[1]*s[1]);
                }
            }
        }

        Mat sq; cv::sqrt(D, sq);
        Mat_<Vec2d> DINV = TOTALPIXEL * size / sq;
        Mat_<Vec2d> DINV_S(TOTALPIXEL, size, 0.0);
        Mat_<Vec2d> SPLUS_DINV(size, TOTALPIXEL, 0.0);
        for (int l=0; l<size; l++) {
            for (int m=0; m<TOTALPIXEL; m++) {
                SPLUS_DINV(l, m)[0] = SPLUS(l,m)[0] * DINV(m,0)[0];
                SPLUS_DINV(l, m)[1] = SPLUS(l,m)[1] * DINV(m,0)[1];
                DINV_S(m, l)[0] = S(m,l)[0] * DINV(m,0)[0];
                DINV_S(m, l)[1] = S(m,l)[1] * DINV(m,0)[1];
            }
        }

        Mat_<Vec2d> SPLUS_DINV_S = SPLUS_DINV * S;
        Mat_<Vec2d> SPLUS_DINV_S_INV(size, size);
        Mat_<double> SPLUS_DINV_S_INV_1(2*size, 2*size);
        for (int l=0; l<size; l++) {
            for (int m=0; m<size; m++) {
                Vec2d s = SPLUS_DINV_S(l, m);
                SPLUS_DINV_S_INV_1(l,      m)      = s[0];
                SPLUS_DINV_S_INV_1(l+size, m+size) = s[0];
                SPLUS_DINV_S_INV_1(l,      m+size) = s[1];
                SPLUS_DINV_S_INV_1(l+size, m)     = -s[1];
            }
        }
        invert(SPLUS_DINV_S_INV_1, SPLUS_DINV_S_INV_1);

        for (int l=0; l<size; l++) {
            for (int m=0; m<size; m++) {
                SPLUS_DINV_S_INV(l, m) = Vec2d(SPLUS_DINV_S_INV_1(l,m), SPLUS_DINV_S_INV_1(l,m+size));
            }
        }

        Mat_<Vec2d> Hmace = DINV_S * SPLUS_DINV_S_INV;
        Mat_<Vec2d> C(size,1, Vec2d(1,0));
        maceFilter = Mat(Hmace * C).reshape(2,IMGSIZE_2X);
    }

    // get the lowest (worst) positive train correlation,
    // our lower bound threshold for the "same()" test later
    double computeThreshold(const std::vector<Mat> &images, bool isdft) const {
        double best=DBL_MAX;
        for (size_t i=0; i<images.size(); i++) {
            double d = correlate(images[i], isdft);
            if (d < best) {
                best = d;
            }
        }
        return best;
    }

    // convolute macefilter and dft image,
    // calculate the peak to sidelobe ratio
    // on the real part of the inverse dft
    double correlate(const Mat &img) const {
        return correlate(img, false);
    }
    double correlate(const Mat &img, bool isdft) const {
        if (maceFilter.empty()) return -1; // not trained.
        int  IMGSIZE_2X = IMGSIZE * 2;
        Mat dftImg = isdft ? img : dftImage(img);
        mulSpectrums(dftImg, maceFilter, dftImg, DFT_ROWS, true);
        dft(dftImg, dftImg, DFT_INVERSE|DFT_SCALE, 0);
        Mat chn[2];
        split(dftImg, chn);
        Mat_<double> re;
        shiftDFT(chn[0], re);
        double m1,M1;
        minMaxLoc(re, &m1, &M1, 0, 0);
        double peakCorrPlaneEnergy = M1 / sqrt(sum(re)[0]);
        re -= m1;
        double value=0;
        double num=0;
        int rad_1=int(floor((double)(45.0/64.0)*(double)IMGSIZE));
        int rad_2=int(floor((double)(27.0/64.0)*(double)IMGSIZE));
        // cache a few pow's and sqrts
        std::vector<double> r2(IMGSIZE_2X);
        Mat_<double> radtab(IMGSIZE_2X,IMGSIZE_2X);
        for (int l=0; l<IMGSIZE_2X; l++) {
            r2[l] = (l-IMGSIZE) * (l-IMGSIZE);
        }
        for (int l=0; l<IMGSIZE_2X; l++) {
            for (int m=l+1; m<IMGSIZE_2X; m++) {
                double rad = sqrt(r2[m] + r2[l]);
                radtab(l,m) = radtab(m,l) = rad;
            }
        }
        // mean of the sidelobe area:
        for (int l=0; l<IMGSIZE_2X; l++) {
            for (int m=0; m<IMGSIZE_2X; m++) {
                double rad = radtab(l,m);
                if (rad < rad_1) {
                    if (rad > rad_2) {
                        value += re(l,m);
                        num++;
                    }
                }
            }
        }
        value /= num;
        // normalize it
        double std2=0;
        for (int l=0; l<IMGSIZE_2X; l++) {
            for (int m=0; m<IMGSIZE_2X; m++) {
                double rad = radtab(l,m);
                if (rad < rad_1) {
                    if (rad > rad_2) {
                        double d = (value - re(l,m));
                        std2 += d * d;
                    }
                }
            }
        }
        std2 /= num;
        std2 = sqrt(std2);
        double sca = re(IMGSIZE, IMGSIZE);
        double peakToSideLobeRatio = (sca - value) / std2;

        return 100.0 * peakToSideLobeRatio * peakCorrPlaneEnergy;
    }

    // MACE interface
    void train(InputArrayOfArrays input) CV_OVERRIDE {
        std::vector<Mat> images, dftImg;
        input.getMatVector(images);
        for (size_t i=0; i<images.size(); i++) { // cache dft images
            dftImg.push_back(dftImage(images[i]));
        }
        compute(dftImg, true);
        threshold = computeThreshold(dftImg, true);
    }
    bool same(InputArray img) const CV_OVERRIDE {
        return correlate(img.getMat()) >= threshold;
    }

    // cv::Algorithm:
    bool empty() const CV_OVERRIDE {
        return maceFilter.empty() || IMGSIZE == 0;
    }
    String  getDefaultName () const CV_OVERRIDE {
        return String("MACE");
    }
    void clear() CV_OVERRIDE {
        maceFilter.release();
        convFilter.release();
    }
    void write(cv::FileStorage &fs) const CV_OVERRIDE {
        fs << "mace" << maceFilter;
        fs << "conv" << convFilter;
        fs << "threshold" << threshold;
    }
    void read(const cv::FileNode &fn) CV_OVERRIDE {
        fn["mace"] >> maceFilter;
        fn["conv"] >> convFilter;
        fn["threshold"] >> threshold;
        IMGSIZE = maceFilter.cols/2;
    }
};


cv::Ptr<MACE> MACE::create(int siz) {
    return makePtr<MACEImpl>(siz);
}
cv::Ptr<MACE> MACE::load(const String &filename, const String &objname) {
    return Algorithm::load<MACE>(filename, objname);
}

} /* namespace face */
} /* namespace cv */
