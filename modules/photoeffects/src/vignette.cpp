#include "precomp.hpp"

namespace cv { namespace photoeffects {

class VignetteInvoker: public cv::ParallelLoopBody
{
public:
    VignetteInvoker(const Mat& src, Mat& dst, Size rect):
        imgSrc(src),
        imgDst(dst)
    {
        centerRow = imgSrc.rows / 2.0f;
        centerCol = imgSrc.cols / 2.0f;
        aSquare = rect.height * rect.height / 4.0f;
        bSquare = rect.width * rect.width / 4.0f;
        distMax = centerRow * centerRow / aSquare + centerCol * centerCol / bSquare - 1.0f;
    }

    void operator()(const Range& rows) const
    {
        Mat srcStripe = imgSrc.rowRange(rows.start, rows.end);
        Mat dstStripe = imgDst.rowRange(rows.start, rows.end);
        srcStripe.copyTo(dstStripe);

        for (int i = rows.start; i < rows.end; i++)
        {
            uchar* dstRow = (uchar*)dstStripe.row(i - rows.start).data;
            for (int j = 0; j < imgSrc.cols; j++)
            {
                float dist = (i - centerRow) * (i - centerRow) / aSquare +
                    (j - centerCol) * (j - centerCol) / bSquare;
                float coefficient = 1.0f;
                if (dist > 1.0f)
                {
                    coefficient = 1.0f - (dist - 1.0f) / distMax;
                }
                dstRow[3 * j] = (uchar)(dstRow[3 * j] * coefficient);
                dstRow[3 * j + 1] = (uchar)(dstRow[3 * j + 1] * coefficient);
                dstRow[3 * j + 2] = (uchar)(dstRow[3 * j + 2] * coefficient);
            }
        }
    }

private:
    const Mat& imgSrc;
    Mat& imgDst;
    float centerRow, centerCol, aSquare, bSquare, distMax;

    VignetteInvoker& operator=(const VignetteInvoker&);
};

void vignette(InputArray src, OutputArray dst, Size rect)
{
    CV_Assert(src.type() == CV_8UC3 && rect.height != 0 && rect.width != 0);

    Mat imgSrc = src.getMat();
    CV_Assert(imgSrc.data);

    dst.create(imgSrc.size(), CV_8UC3);
    Mat imgDst = dst.getMat();

    parallel_for_(Range(0, imgSrc.rows), VignetteInvoker(imgSrc, imgDst, rect));
}

}}
