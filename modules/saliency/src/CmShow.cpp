#include "kyheader.h"
#include "CmShow.h"
#include "opencv2/core.hpp"
#include <opencv2/highgui.hpp>



typedef pair<int, int> CostiIdx;
Mat CmShow::HistBins(CMat& color3f, CMat& val, CStr& title, bool descendShow, CMat &with)
{
    // Prepare data
    int H = 300, spaceH = 6, barH = 10, n = color3f.cols;
    CV_Assert(color3f.size() == val.size() && color3f.rows == 1);
    Mat binVal1i, binColor3b, width1i;
    if (with.size() == val.size())
        with.convertTo(width1i, CV_32S, 400/sum(with).val[0]); // Default shown width
    else
        width1i = Mat(1, n, CV_32S, Scalar(10)); // Default bin width = 10
    int W = cvRound(sum(width1i).val[0]);
    color3f.convertTo(binColor3b, CV_8UC3, 255);
    double maxVal, minVal;
    minMaxLoc(val, &minVal, &maxVal);
    printf("%g\n", H/max(maxVal, -minVal));
    val.convertTo(binVal1i, CV_32S, 20000);
    Size szShow(W, H + spaceH + barH);
    szShow.height += minVal < 0 && !descendShow ? H + spaceH : 0;
    Mat showImg3b(szShow, CV_8UC3, Scalar(255, 255, 255));
    int* binH = (int*)(binVal1i.data);
    Vec3b* binColor = (Vec3b*)(binColor3b.data);
    int* binW = (int*)(width1i.data);
    vector<CostiIdx> costIdx(n);
    if (descendShow){
        for (int i = 0; i < n; i++)
            costIdx[i] = make_pair(binH[i], i);
        sort(costIdx.begin(), costIdx.end(), std::greater<CostiIdx>());
    }

    // Show image
    for (int i = 0, x = 0; i < n; i++){
        int idx = descendShow ? costIdx[i].second : i;
        int h = descendShow ? abs(binH[idx]) : binH[idx];
        Scalar color(binColor[idx]);
        Rect reg(x, H + spaceH, binW[idx], barH);
        showImg3b(reg) = color; // Draw bar
        rectangle(showImg3b, reg, Scalar(0));

        reg.height = abs(h);
        reg.y = h >= 0 ? H - h : H + 2 * spaceH + barH;
        showImg3b(reg) = color;
        rectangle(showImg3b, reg, Scalar(0));

        x += binW[idx];
    }
    imshow(String(title.c_str()), showImg3b);
    return showImg3b;
}

void CmShow::showTinyMat(CStr &title, CMat &m)
{
    int scale = 50, sz = m.rows * m.cols;
    while (sz > 200){
        scale /= 2;
        sz /= 4;
    }

    Mat img;
    resize(m, img, Size(), scale, scale, INTER_NEAREST );
    if (img.channels() == 3)
        cvtColor(img, img, COLOR_RGB2BGR);
    SaveShow(img, title);
}

void CmShow::SaveShow(CMat& img, CStr& title)
{
    if (title.size() == 0)
        return;

    int mDepth = CV_MAT_DEPTH(img.type());
    double scale = (mDepth == CV_32F || mDepth == CV_64F ? 255 : 1);
    if (title.size() > 4 && title[title.size() - 4] == '.')
        imwrite(String(title.c_str()), img*scale);
    else if (title.size())
        imshow(String(title.c_str()), img);
}
