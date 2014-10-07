#include "precomp.hpp"

namespace cv { namespace photoeffects {

class edgeBlurInvoker :public ParallelLoopBody
{
public:
    edgeBlurInvoker(const Mat& src,
                    const Mat& boxFilt,
                    Mat& dst,
                    int indentTop,
                    int indentLeft)
    :   src_(src),
        boxFilt_(boxFilt),
        dst_(dst),
        indentTop_(indentTop),
        indentLeft_(indentLeft) {}

    void operator()(const Range& range) const
    {
        float halfWidth = src_.cols / 2.0f;
        float halfHeight = src_.rows / 2.0f;
        float a = (halfWidth - indentLeft_) * (halfWidth - indentLeft_);
        float b = (halfHeight - indentTop_) * (halfHeight - indentTop_);

        Mat srcStripe = src_.rowRange(range.start, range.end);
        Mat boxStripe = boxFilt_.rowRange(range.start, range.end);
        Mat dstStripe = dst_.rowRange(range.start, range.end);

        int rows = srcStripe.rows;
        for (int i = 0; i < rows; i++)
        {
            uchar* row = (uchar*)srcStripe.row(i).data;
            uchar* boxRow = (uchar*)boxStripe.row(i).data;
            uchar* dstRow = (uchar*)dstStripe.row(i).data;
            float y_part = (halfHeight - (i + range.start)) *
                           (halfHeight - (i + range.start)) / b;

            for (int j = 0; j < 3 * src_.cols; j += 3)
            {
                float maskEl = min(max(2.0f *
                               ((halfWidth - j / 3) * (halfWidth - j / 3) / a +
                               y_part - 0.5f), 0.0f), 1.0f);
                float negMask = 1.0f - maskEl;

                dstRow[j] = boxRow[j] * maskEl + row[j] * negMask;
                dstRow[j + 1] = boxRow[j + 1] * maskEl + row[j + 1] * negMask;
                dstRow[j + 2] = boxRow[j + 2] * maskEl + row[j + 2] * negMask;
            }
        }
    }

private:
    const Mat& src_;
    const Mat& boxFilt_;
    Mat& dst_;
    int indentTop_;
    int indentLeft_;

    edgeBlurInvoker& operator=(const edgeBlurInvoker&);
};

void edgeBlur(InputArray src, OutputArray dst, int indentTop, int indentLeft)
{
    CV_Assert(!src.empty());
    CV_Assert(src.type() == CV_8UC3);

    dst.create(src.size(), src.type());
    Mat image = src.getMat(), outputImage = dst.getMat();

    CV_Assert(indentTop >= 0 && indentTop <= (image.rows / 2 - 10));
    CV_Assert(indentLeft >= 0 && indentLeft <= (image.cols / 2 - 10));

    Mat boxFilt;

    boxFilter(image, boxFilt, -1, Size(7, 7), Point(-1, -1),
        true, BORDER_REPLICATE);

    parallel_for_(Range(0, image.rows),
                  edgeBlurInvoker(image,
                                  boxFilt,
                                  outputImage,
                                  indentTop,
                                  indentLeft));
}

}}
