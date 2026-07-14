// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <limits>
#include <utility>
#include <vector>

namespace cv {
namespace ximgproc {
namespace stMorph {

// normalizeAnchor; Copied from filterengine.hpp.
static inline Point normalizeAnchor(Point anchor, Size ksize)
{
    if (anchor.x == -1)
        anchor.x = ksize.width / 2;
    if (anchor.y == -1)
        anchor.y = ksize.height / 2;
    CV_Assert(anchor.inside(Rect(0, 0, ksize.width, ksize.height)));
    return anchor;
}

static int log2(int n)
{
    int ans = -1;
    while (n > 0)
    {
        n /= 2;
        ans++;
    }
    return ans;
}

static int longestRowRunLength(const Mat& kernel)
{
    int cnt = 0;
    int maxLen = 0;
    for (int c = 0; c < kernel.cols; c++)
    {
        cnt = 0;
        for (int r = 0; r < kernel.rows; r++)
        {
            if (kernel.at<uchar>(r, c) == 0)
            {
                maxLen = std::max(maxLen, cnt);
                cnt = 0;
            }
            else cnt++;
        }
        maxLen = std::max(maxLen, cnt);
    }
    return maxLen;
}

static int longestColRunLength(const Mat& kernel)
{
    int cnt = 0;
    int maxLen = 0;
    for (int r = 0; r < kernel.rows; r++)
    {
        cnt = 0;
        for (int c = 0; c < kernel.cols; c++)
        {
            if (kernel.at<uchar>(r, c) == 0)
            {
                maxLen = std::max(maxLen, cnt);
                cnt = 0;
            }
            else cnt++;
        }
        maxLen = std::max(maxLen, cnt);
    }
    return maxLen;
}

static std::vector<Point> findP2RectCorners(const Mat& stNode, int rowDepth, int colDepth)
{
    int rowOfst = 1 << rowDepth;
    int colOfst = 1 << colDepth;
    std::vector<Point> p2Rects;
    for (int row = 0; row < stNode.rows; row++)
    {
        for (int col = 0; col < stNode.cols; col++)
        {
            // select white cells
            if (stNode.at<uchar>(row, col) == 0) continue;

            // select corner cells
            if (col > 0 && stNode.at<uchar>(row, col - 1) == 1
                && col + 1 < stNode.cols && stNode.at<uchar>(row, col + 1) == 1) continue;
            if (row > 0 && stNode.at<uchar>(row - 1, col) == 1
                && row + 1 < stNode.rows && stNode.at<uchar>(row + 1, col) == 1) continue;

            // ignore if deeper cell is white
            if (col + colOfst < stNode.cols && stNode.at<uchar>(row, col + colOfst) == 1) continue;
            if (col - colOfst >= 0 && stNode.at<uchar>(row, col - colOfst) == 1) continue;
            if (row + rowOfst < stNode.rows && stNode.at<uchar>(row + rowOfst, col) == 1) continue;
            if (row - rowOfst >= 0 && stNode.at<uchar>(row - rowOfst, col) == 1) continue;

            p2Rects.emplace_back(col, row);
        }
    }
    return p2Rects;
}

/*
* Find a set of power-2-rectangles to cover the kernel.
* power-2-rectangles is a rectangle whose height and width are both power of 2.
*/
static std::vector<std::vector<std::vector<Point>>> genPow2RectsToCoverKernel(
    const Mat& kernel, int rowDepthLim, int colDepthLim)
{
    CV_Assert(kernel.type() == CV_8UC1);

    std::vector<std::vector<std::vector<Point>>> p2Rects;
    Mat stNodeCache = kernel;
    for (int rowDepth = 0; rowDepth < rowDepthLim; rowDepth++)
    {
        Mat stNode = stNodeCache.clone();
        p2Rects.emplace_back(std::vector<std::vector<Point>>());
        for (int colDepth = 0; colDepth < colDepthLim; colDepth++)
        {
            p2Rects[rowDepth].emplace_back(findP2RectCorners(stNode, rowDepth, colDepth));
            int colStep = 1 << colDepth;
            if (stNode.cols - colStep < 0) break;
            Rect s1(0, 0, stNode.cols - colStep, stNode.rows);
            Rect s2(colStep, 0, stNode.cols - colStep, stNode.rows);
            cv::min(stNode(s1), stNode(s2), stNode);
        }
        int rowStep = 1 << rowDepth;
        if (stNodeCache.rows - rowStep < 0) break;
        Rect s1(0, 0, stNodeCache.cols, stNodeCache.rows - rowStep);
        Rect s2(0, rowStep, stNodeCache.cols, stNodeCache.rows - rowStep);
        cv::min(stNodeCache(s1), stNodeCache(s2), stNodeCache);
    }

    return p2Rects;
}

/*
* Solves the rectilinear steiner arborescence problem greedy.
*/
static Mat SolveRSAPGreedy(const Mat& initialMap)
{
    CV_Assert(initialMap.type() == CV_8UC1);
    std::vector<Point> pos;
    for (int r = 0; r < initialMap.rows; r++)
        for (int c = 0; c < initialMap.cols; c++)
            if (initialMap.at<uchar>(r, c) == 1) pos.emplace_back(c, r);
    Mat resMap = Mat::zeros(initialMap.size(), CV_8UC2);

    while (pos.size() > 1)
    {
        int maxCost = -1;
        int maxI = 0;
        int maxJ = 0;
        int maxX = 0;
        int maxY = 0;
        for (uint i = 0; i < pos.size(); i++)
        {
            for (uint j = i + 1; j < pos.size(); j++)
            {
                int _x = std::min(pos[i].x, pos[j].x);
                int _y = std::min(pos[i].y, pos[j].y);
                int cost = _x + _y;
                if (maxCost < cost)
                {
                    maxCost = cost;
                    maxI = i;
                    maxJ = j;
                    maxX = _x;
                    maxY = _y;
                }
            }
        }
        for (int col = pos[maxI].x - 1; col >= maxX; col--)
            resMap.at<Vec2b>(pos[maxI].y, col)[1] = 1;
        for (int row = pos[maxI].y - 1; row >= maxY; row--)
            resMap.at<Vec2b>(row, maxX)[0] = 1;
        for (int col = pos[maxJ].x - 1; col >= maxX; col--)
            resMap.at<Vec2b>(pos[maxJ].y, col)[1] = 1;
        for (int row = pos[maxJ].y - 1; row >= maxY; row--)
            resMap.at<Vec2b>(row, maxX)[0] = 1;

        pos[maxI] = Point(maxX, maxY);
        std::swap(pos[maxJ], pos[pos.size() - 1]);
        pos.pop_back();
    }
    return resMap;
}

static Mat sparseTableFillPlanning(
    std::vector<std::vector<std::vector<Point>>> pow2Rects, int rowDepthLim, int colDepthLim)
{
    // list up required sparse table nodes.
    Mat stMap = Mat::zeros(rowDepthLim, colDepthLim, CV_8UC1);
    for (int rd = 0; rd < rowDepthLim; rd++)
        for (int cd = 0; cd < colDepthLim; cd++)
            if (pow2Rects[rd][cd].size() > 0)
                stMap.at<uchar>(rd, cd) = 1;
    stMap.at<uchar>(0, 0) = 1;
    Mat path = SolveRSAPGreedy(stMap);
    return path;
}

kernelDecompInfo decompKernel(InputArray kernel, Point anchor, int iterations)
{
    Mat _kernel = kernel.getMat();
    // Fix kernel in case of it is empty.
    if (_kernel.empty())
    {
        _kernel = getStructuringElement(MORPH_RECT, Size(1 + iterations * 2, 1 + iterations * 2));
        anchor = Point(iterations, iterations);
        iterations = 1;
    }
    if (countNonZero(_kernel) == 0)
    {
        _kernel.at<uchar>(0, 0) = 1;
    }

    int rowDepthLim = log2(longestRowRunLength(_kernel)) + 1;
    int colDepthLim = log2(longestColRunLength(_kernel)) + 1;
    std::vector<std::vector<std::vector<Point>>> pow2Rects
        = genPow2RectsToCoverKernel(_kernel, rowDepthLim, colDepthLim);

    Mat stPlan
        = sparseTableFillPlanning(pow2Rects, rowDepthLim, colDepthLim);

    // Fix anchor to the center of the kernel.
    anchor = stMorph::normalizeAnchor(anchor, _kernel.size());

    return { _kernel.rows, _kernel.cols, pow2Rects, stPlan, anchor, iterations };
}

enum Op
{
    Min, Max
};

static void morphDfs(int minmax, Mat& st, Mat& dst,
    std::vector<std::vector<std::vector<Point>>> row2Rects, const Mat& stPlan,
    int rowDepth, int colDepth)
{
    for (Point p : row2Rects[rowDepth][colDepth])
    {
        Rect rect(p, dst.size());
        if (minmax == Op::Min) cv::min(dst, st(rect), dst);
        else cv::max(dst, st(rect), dst);
    }

    if (stPlan.at<Vec2b>(rowDepth, colDepth)[1] == 1)
    {
        // col direction
        Mat st2 = st;
        int ofs = 1 << colDepth;
        Rect rect1(0, 0, st2.cols - ofs, st2.rows);
        Rect rect2(ofs, 0, st2.cols - ofs, st2.rows);

        if (minmax == Op::Min) cv::min(st2(rect1), st2(rect2), st2);
        else cv::max(st2(rect1), st2(rect2), st2);
        morphDfs(minmax, st2, dst, row2Rects, stPlan, rowDepth, colDepth + 1);
    }
    if (stPlan.at<Vec2b>(rowDepth, colDepth)[0] == 1)
    {
        // row direction
        int ofs = 1 << rowDepth;
        Rect rect1(0, 0, st.cols, st.rows - ofs);
        Rect rect2(0, ofs, st.cols, st.rows - ofs);

        if (minmax == Op::Min) cv::min(st(rect1), st(rect2), st);
        else cv::max(st(rect1), st(rect2), st);
        morphDfs(minmax, st, dst, row2Rects, stPlan, rowDepth + 1, colDepth);
    }
}

template <typename T>
static void morphOp(Op minmax, InputArray _src, OutputArray _dst, kernelDecompInfo kdi,
    BorderTypes borderType, const Scalar& borderVal)
{
    T nil = (minmax == Op::Min) ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();

    Mat src = _src.getMat();
    _dst.create(_src.size(), _src.type());
    Mat dst = _dst.getMat();

    Scalar bV = borderVal;
    if (borderType == BorderTypes::BORDER_CONSTANT && borderVal == morphologyDefaultBorderValue())
        bV = Scalar::all(nil);

    do
    {
        Mat expandedSrc;
        copyMakeBorder(src, expandedSrc,
            kdi.anchor.y, kdi.cols - 1 - kdi.anchor.y,
            kdi.anchor.x, kdi.rows - 1 - kdi.anchor.x,
            borderType, bV);
        dst.setTo(nil);
        morphDfs(minmax, expandedSrc, dst, kdi.stRects, kdi.plan, 0, 0);
        src = dst;
    } while (--kdi.iterations > 0);
}

static void morphOp(Op minmax, InputArray _src, OutputArray _dst, kernelDecompInfo kdi,
    BorderTypes borderType, const Scalar& borderVal)
{
    if (kdi.iterations == 0 || kdi.rows * kdi.cols == 1)
    {
        _src.copyTo(_dst);
        return;
    }

    switch (_src.depth())
    {
    case CV_8U:
        morphOp<uchar>(minmax, _src, _dst, kdi, borderType, borderVal);
        return;
    case CV_8S:
        morphOp<char>(minmax, _src, _dst, kdi, borderType, borderVal);
        return;
    case CV_16U:
        morphOp<ushort>(minmax, _src, _dst, kdi, borderType, borderVal);
        return;
    case CV_16S:
        morphOp<short>(minmax, _src, _dst, kdi, borderType, borderVal);
        return;
    case CV_32S:
        morphOp<int>(minmax, _src, _dst, kdi, borderType, borderVal);
        return;
    case CV_32F:
        morphOp<float>(minmax, _src, _dst, kdi, borderType, borderVal);
        return;
    case CV_64F:
        morphOp<double>(minmax, _src, _dst, kdi, borderType, borderVal);
        return;
    }
}

void erode(InputArray src, OutputArray dst, kernelDecompInfo kdi,
    BorderTypes borderType, const Scalar& borderVal)
{
    morphOp(Op::Min, src, dst, kdi, borderType, borderVal);
}

void dilate(InputArray src, OutputArray dst, kernelDecompInfo kdi,
    BorderTypes borderType, const Scalar& borderVal)
{
    morphOp(Op::Max, src, dst, kdi, borderType, borderVal);
}

void morphologyEx(InputArray src, OutputArray dst, int op, kernelDecompInfo kdi,
    BorderTypes borderType, const Scalar& borderVal)
{
    CV_INSTRUMENT_REGION();

    CV_Assert(!src.empty());

    Mat _src = src.getMat(), temp;
    dst.create(_src.size(), _src.type());
    Mat _dst = dst.getMat();

    switch (op)
    {
    case MORPH_ERODE:
        erode(src, dst, kdi, borderType, borderVal);
        break;
    case MORPH_DILATE:
        dilate(src, dst, kdi, borderType, borderVal);
        break;
    case MORPH_OPEN:
        stMorph::erode(src, dst, kdi, borderType, borderVal);
        stMorph::dilate(dst, dst, kdi, borderType, borderVal);
        break;
    case MORPH_CLOSE:
        stMorph::dilate(src, dst, kdi, borderType, borderVal);
        stMorph::erode(dst, dst, kdi, borderType, borderVal);
        break;
    case MORPH_GRADIENT:
        stMorph::erode(_src, temp, kdi, borderType, borderVal);
        stMorph::dilate(_src, _dst, kdi, borderType, borderVal);
        _dst -= temp;
        break;
    case MORPH_TOPHAT:
        if (_src.data != _dst.data)
            temp = _dst;
        stMorph::erode(_src, temp, kdi, borderType, borderVal);
        stMorph::dilate(temp, temp, kdi, borderType, borderVal);
        _dst = _src - temp;
        break;
    case MORPH_BLACKHAT:
        if (_src.data != _dst.data)
            temp = _dst;
        stMorph::dilate(_src, temp, kdi, borderType, borderVal);
        stMorph::erode(temp, temp, kdi, borderType, borderVal);
        _dst = temp - _src;
        break;
    case MORPH_HITMISS:
        CV_Error(cv::Error::StsBadArg, "HIT-MISS operation is not supported.");
    default:
        CV_Error(cv::Error::StsBadArg, "Unknown morphological operation.");
    }
}

void erode(InputArray src, OutputArray dst, InputArray kernel,
    Point anchor, int iterations,
    BorderTypes borderType, const Scalar& borderVal)
{
    kernelDecompInfo kdi = decompKernel(kernel, anchor, iterations);
    morphOp(Op::Min, src, dst, kdi, borderType, borderVal);
}

void dilate(InputArray src, OutputArray dst, InputArray kernel,
    Point anchor, int iterations,
    BorderTypes borderType, const Scalar& borderVal)
{
    kernelDecompInfo kdi = decompKernel(kernel, anchor, iterations);
    morphOp(Op::Max, src, dst, kdi, borderType, borderVal);
}

void morphologyEx(InputArray src, OutputArray dst, int op,
    InputArray kernel, Point anchor, int iterations,
    BorderTypes borderType, const Scalar& borderVal)
{
    kernelDecompInfo kdi = decompKernel(kernel, anchor, iterations);
    morphologyEx(src, dst, op, kdi, borderType, borderVal);
}

}}} // cv::ximgproc::stMorph::
