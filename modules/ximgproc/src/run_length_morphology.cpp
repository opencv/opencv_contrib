/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */
#include "precomp.hpp"
#include <math.h>
#include <vector>
#include <iostream>


namespace cv {
namespace ximgproc {
namespace rl {

struct rlType
{
  int cb, ce, r;
  rlType(int cbIn, int ceIn, int rIn): cb(cbIn), ce(ceIn), r(rIn) {}
  rlType(): cb(0), ce(0), r(0) {}
  bool operator < (const rlType& other) const { if (r < other.r || (r == other.r && cb < other.cb)
      || (r == other.r && cb == other.cb && ce < other.ce)) return true; else return false; }
};

typedef std::vector<rlType> rlVec;

template <class T>
void _thresholdLine(T* pData, int nWidth, int nRow, T threshold, int type, rlVec& res)
{
  bool bOn = false;
  int nStartSegment = 0;
  for (int j = 0; j < nWidth; j++)
  {
    bool bAboveThreshold = (pData[j] > threshold);
    bool bCurOn = (bAboveThreshold == (THRESH_BINARY == type));
    if (!bOn && bCurOn)
    {
      nStartSegment = j;
      bOn = true;
    }
    else if (bOn && !bCurOn)
    {
      rlType chord(nStartSegment, j - 1, nRow);
      res.push_back(chord);
      bOn = false;
    }

  }
  if (bOn)
  {
    rlType chord(nStartSegment, nWidth - 1, nRow);
    res.push_back(chord);
  }
}

static void _threshold(cv::Mat& img, rlVec& res, double threshold, int type)
{
  res.clear();
  switch (img.depth())
  {
  case CV_8U:
    for (int i = 0; i < img.rows; ++i)
      _thresholdLine<uchar>(img.ptr(i), img.cols, i, (uchar) threshold, type, res);
    break;
  case CV_8S:
    for (int i = 0; i < img.rows; ++i)
      _thresholdLine<schar>((schar*) img.ptr(i), img.cols, i, (schar) threshold, type, res);
    break;
  case CV_16U:
      for (int i = 0; i < img.rows; ++i)
      {
          _thresholdLine<unsigned short>((unsigned short*)img.ptr(i), img.cols, i,
              (unsigned short)threshold, type, res);
      }
    break;
  case CV_16S:
    for (int i = 0; i < img.rows; ++i)
      _thresholdLine<short>((short*) img.ptr(i), img.cols, i, (short) threshold, type, res);
    break;
  case CV_32S:
    for (int i = 0; i < img.rows; ++i)
      _thresholdLine<int>((int*) img.ptr(i), img.cols, i, (int) threshold, type, res);
    break;
  case CV_32F:
    for (int i = 0; i < img.rows; ++i)
      _thresholdLine<float>((float*) img.ptr(i), img.cols, i, (float) threshold, type, res);
    break;
  case CV_64F:
    for (int i = 0; i < img.rows; ++i)
      _thresholdLine<double>((double*) img.ptr(i), img.cols, i, threshold, type, res);
    break;
  default:
    CV_Error( Error::StsUnsupportedFormat, "unsupported image type" );
  }
}


static void convertToOutputArray(rlVec& runs, Size size, OutputArray& res)
{
    size_t nRuns = runs.size();
    std::vector<cv::Point3i> segments(nRuns + 1);
    segments[0] = cv::Point3i(size.width, size.height, 0);
    for (size_t i = 0; i < nRuns; ++i)
    {
        rlType& curRun = runs[i];
        segments[i + 1] = Point3i(curRun.cb, curRun.ce, curRun.r);
    }
    Mat(segments).copyTo(res);
}


CV_EXPORTS void threshold(InputArray src, OutputArray rlDest, double thresh, int type)
{
    CV_INSTRUMENT_REGION();

    Mat image = src.getMat();
    CV_Assert(!image.empty() && image.channels() == 1);
    CV_Assert(type == THRESH_BINARY || type == THRESH_BINARY_INV);
    rlVec runs;
    _threshold(image, runs, thresh, type);
    Size size(image.cols, image.rows);

    convertToOutputArray(runs, size, rlDest);
}


template <class T>
void paint_impl(cv::Mat& img, rlType* pRuns, int nSize, T value)
{
    int i;
    rlType* pCurRun;
    for (pCurRun = pRuns, i = 0; i< nSize; ++pCurRun, ++i)
    {
        rlType& curRun = *pCurRun;
        if (curRun.r < 0 || curRun.r >= img.rows || curRun.cb >= img.cols || curRun.ce < 0)
            continue;

        T* rowPtr = (T*)img.ptr(curRun.r);
        std::fill(rowPtr + std::max(curRun.cb, 0), rowPtr + std::min(curRun.ce + 1, img.cols), value);
    }
}

  CV_EXPORTS void paint(InputOutputArray image, InputArray rlSrc, const Scalar& value)
  {
    Mat _runs;
    _runs = rlSrc.getMat();
    int N = _runs.checkVector(3);
    if (N <= 1)
        return;

    double dValue = value[0];

    cv::Mat _image = image.getMat();

    rlType* pRuns = (rlType*) &(_runs.at<Point3i>(1));
    switch (_image.type())
    {
    case CV_8UC1:
        paint_impl<uchar>(_image, pRuns, N - 1, (uchar)dValue);
        break;
    case CV_8SC1:
        paint_impl<schar>(_image, pRuns, N - 1, (schar)dValue);
        break;
    case CV_16UC1:
        paint_impl<unsigned short>(_image, pRuns, N - 1, (unsigned short)dValue);
        break;
    case CV_16SC1:
        paint_impl<short>(_image, pRuns, N - 1, (short)dValue);
        break;
    case CV_32SC1:
        paint_impl<int>(_image, pRuns, N - 1, (int)dValue);
        break;
    case CV_32FC1:
        paint_impl<float>(_image, pRuns, N - 1, (float)dValue);
        break;
    case CV_64FC1:
        paint_impl<double>(_image, pRuns, N - 1, dValue);
        break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "unsupported image type");
        break;
    }
  }

static void translateRegion(rlVec& reg, Point ptTrans)
{
    for (rlVec::iterator it=reg.begin();it!=reg.end();++it)
    {
        it->r += ptTrans.y;
        it->cb += ptTrans.x;
        it->ce += ptTrans.x;
    }
}

CV_EXPORTS Mat getStructuringElement(int shape, Size ksize)
{
  Mat mask = cv::getStructuringElement(shape, ksize);

  rlVec reg;
  _threshold(mask, reg, 0.0, THRESH_BINARY);

  Point ptTrans = - Point(mask.cols / 2, mask.rows / 2);
  translateRegion(reg, ptTrans);
  Mat rlDest;
  convertToOutputArray(reg, Size(mask.cols, mask.rows), rlDest);

  return rlDest;
}

static void erode_rle (rlVec& regIn, rlVec& regOut, rlVec& se)
{
  using namespace std;

    regOut.clear();

    if (regIn.size() == 0)
        return;

    int nMinRow = regIn[0].r;
    int nMaxRow = regIn.back().r;

    int nRows = nMaxRow - nMinRow + 1;


    const int EMPTY = -1;

    // setup a table which holds the index of the first chord for each row
    vector<int> pIdxChord1(nRows);
    vector<int> pIdxNextRow(nRows);

    int i,j;

    for (i=1;i<nRows;i++)
    {
        pIdxChord1[i] = EMPTY;
        pIdxNextRow[i] = EMPTY;
    }

    pIdxChord1[0] = 0;
    pIdxNextRow[nRows-1] = (int) regIn.size();

    for (i=1; i < (int) regIn.size();i++)
        if (regIn[i].r != regIn[i-1].r)
        {
            pIdxChord1[regIn[i].r - nMinRow] = i;
            pIdxNextRow[regIn[i-1].r - nMinRow] = i;
        }

    int nMinRowSE = se[0].r;
    int nMaxRowSE = se.back().r;

    int nRowsSE = nMaxRowSE - nMinRowSE + 1;

    assert(nRowsSE == (int) se.size());

    vector<int> pCurIdxRow(nRowsSE);

    // loop through all possible rows
    for (i=nMinRow - nMinRowSE; i<= nMaxRow - nMaxRowSE; i++)
    {
        // check whether all relevant rows are available
        bool bNextRow = false;

        for (j=0; j < nRowsSE; j++)
        {
            // get idx of first chord in regIn for this row of the se
            pCurIdxRow[j] = pIdxChord1[ j + nMinRowSE + i - nMinRow];
            if (pCurIdxRow[j] == -1)
            {
                bNextRow = true;
                break;
            }
        }

        if (bNextRow)
            continue;

        while (!bNextRow)
        {
          int nPossibleStart = std::numeric_limits<int>::min();

          // search for row with max( cb - se.cb) (the leftmost possible position of a result chord
          for (j=0;j<nRowsSE;j++)
              nPossibleStart = max(nPossibleStart, regIn[pCurIdxRow[j]].cb - se[j].cb);

          // for all rows skip chords whose end is left from the point
          // where it can contribute to a result
          bool bHaveResult = true;
          int nLimitingRow = 0;
          int nChordEnd = std::numeric_limits<int>::max(); //INT_MAX;

          for (j=0;j<nRowsSE;j++)
          {
              while (regIn[pCurIdxRow[j]].ce < nPossibleStart + se[j].ce &&
                  pCurIdxRow[j] != pIdxNextRow[j + nMinRowSE + i - nMinRow])
              {
                  pCurIdxRow[j]++;
              }

              // if all chords in this row skipped -> next row
              if (pCurIdxRow[j] == pIdxNextRow[ j + nMinRowSE + i - nMinRow])
              {
                  bNextRow = true;
                  bHaveResult = false;
                  break;
              }
              else if ( bHaveResult )
              {
              // can the found chord contribute to a result ?
              if (regIn[ pCurIdxRow[j] ].cb - se[j].cb <= nPossibleStart)
              {
                  int nCurPossibleEnd = regIn[ pCurIdxRow[j] ].ce - se[j].ce;
                  if (nCurPossibleEnd < nChordEnd)
                  {
                      nChordEnd = nCurPossibleEnd;
                      nLimitingRow = j;
                  }
              }
              else
                  bHaveResult = false;
              }
          }

        if (bHaveResult)
        {
            regOut.push_back(rlType(nPossibleStart, nChordEnd, i));
            pCurIdxRow[nLimitingRow]++;

            if (pCurIdxRow[nLimitingRow] == pIdxNextRow[ nLimitingRow + nMinRowSE + i - nMinRow])
                  bNextRow = true;
        }
        } // end while (!bNextRow
    } // end for

}

static void convertInputArrayToRuns(InputArray& theArray, rlVec& runs, Size& theSize)
{
  Mat _runs;
  _runs = theArray.getMat();
  int N = _runs.checkVector(3);
  if (N == 0)
  {
      runs.clear();
      return;
  }
  runs.resize(N - 1);
  Point3i pt = _runs.at<Point3i>(0);
  theSize.width = pt.x;
  theSize.height = pt.y;

  for (int i = 1; i < N; ++i)
  {
    pt = _runs.at<Point3i>(i);
    runs[i-1] = rlType(pt.x, pt.y, pt.z);
  }
}

static void sortChords(rlVec& lChords)
{
    sort(lChords.begin(), lChords.end());
}

static void mergeNeighbouringChords(rlVec& rlIn, rlVec& rlOut)
{
    rlOut.clear();
    if (rlIn.size() == 0)
        return;

    rlOut.push_back(rlIn[0]);

    for (int i = 1; i< (int)rlIn.size(); i++)
    {
        rlType& curIn = rlIn[i];
        rlType& lastAddedOut = rlOut.back();
        if (curIn.r == lastAddedOut.r && curIn.cb <= lastAddedOut.ce + 1)
            lastAddedOut.ce = max(curIn.ce, lastAddedOut.ce);
        else
            rlOut.push_back(curIn);
    }
}

static void union_regions(rlVec& reg1, rlVec& reg2, rlVec& regUnion)
{
    rlVec lAllChords(reg1);

    lAllChords.insert(lAllChords.end(), reg2.begin(), reg2.end());

    sortChords(lAllChords);
    mergeNeighbouringChords(lAllChords, regUnion);
}

static void intersect(rlVec& reg1, rlVec& reg2, rlVec& regRes)
{
    rlVec::iterator end1 = reg1.end();
    rlVec::iterator end2 = reg2.end();

    rlVec::iterator cur1 = reg1.begin();
    rlVec::iterator cur2 = reg2.begin();
    regRes.clear();

    while (cur1 != end1 && cur2 != end2)
    {
        if (cur1->r < cur2->r || (cur1->r == cur2->r && cur1->ce < cur2->cb))
            ++cur1;
        else if (cur2->r < cur1->r || (cur1->r == cur2->r && cur2->ce < cur1->cb))
            ++cur2;
        else
        {
            assert(cur1->r == cur2->r);
            int nStart = max(cur1->cb, cur2->cb);
            int nEnd = min(cur1->ce, cur2->ce);
            if (nStart > nEnd)
            {
                assert(nStart <= nEnd);
            }
            regRes.push_back(rlType(nStart, nEnd, cur1->r));
            if (cur1->ce < cur2->ce)
                ++cur1;
            else
                ++cur2;
        }

    }
}

static void addBoundary(rlVec& runsIn, int nWidth, int nHeight, int nBoundaryLeft, int nBoundaryTop,
    int nBoundaryRight, int nBoundaryBottom, rlVec& res)
{
    rlVec boundary;
    for (int i = -nBoundaryTop; i < 0; ++i)
        boundary.push_back(rlType(-nBoundaryLeft, nWidth - 1 + nBoundaryRight, i));
    for (int i = 0; i < nHeight; ++i)
    {
        boundary.push_back(rlType(-nBoundaryLeft, -1, i));
        boundary.push_back(rlType(nWidth, nWidth - 1 + nBoundaryRight, i));
    }

    for (int i = nHeight; i < nHeight + nBoundaryBottom; ++i)
        boundary.push_back(rlType(-nBoundaryLeft, nWidth - 1 + nBoundaryRight, i));

    union_regions(runsIn, boundary, res);
}

static cv::Rect getBoundingRectangle(rlVec& reg)
{
    using namespace std;
    cv::Rect rect;
    if (reg.empty())
    {
        rect.x = rect.y = rect.width = rect.height = 0;
        return rect;
    }

    int minX = std::numeric_limits<int>::max();
    int minY = std::numeric_limits<int>::max();
    int maxX = std::numeric_limits<int>::min();
    int maxY = std::numeric_limits<int>::min();

    int i;
    for (i = 0; i<(int)reg.size(); i++)
    {
        minX = min(minX, reg[i].cb);
        maxX = max(maxX, reg[i].ce);
        minY = min(minY, reg[i].r);
        maxY = max(maxY, reg[i].r);
    }

    rect.x = minX;
    rect.y = minY;
    rect.width = maxX - minX + 1;
    rect.height = maxY - minY + 1;
    return rect;
}

static void createUprightRectangle(cv::Rect rect, rlVec &rl)
{
    rl.clear();
    rlType curRL;
    int j;
    int cb = rect.x;
    int ce = rect.x + rect.width - 1;
    for (j = 0; j < rect.height; j++)
    {
        curRL.cb = cb;
        curRL.ce = ce;
        curRL.r = j + rect.y;
        rl.push_back(curRL);
    }
}

static void erode_with_boundary_rle(rlVec& runsSource, int nWidth, int nHeight, rlVec& runsDestination,
    rlVec& runsKernel)
{
    cv::Rect rect = getBoundingRectangle(runsKernel);
    rlVec regExtended, regFrame, regResultRaw;
    addBoundary(runsSource, nWidth, nHeight, max(0, -rect.x), max(0, -rect.y),
        max(0, rect.x + rect.width), max(0, rect.y + rect.height), regExtended);

    erode_rle(regExtended, regResultRaw, runsKernel);
    createUprightRectangle(cv::Rect(0, 0, nWidth, nHeight), regFrame);
    intersect(regResultRaw, regFrame, runsDestination);
}


CV_EXPORTS void erode(InputArray rlSrc, OutputArray rlDest, InputArray rlKernel, bool bBoundaryOn,
    Point anchor)
{
  rlVec runsSource, runsDestination, runsKernel;
  Size sizeSource, sizeKernel;
  convertInputArrayToRuns(rlSrc, runsSource, sizeSource);
  convertInputArrayToRuns(rlKernel, runsKernel, sizeKernel);

  if (anchor != Point(0,0))
    translateRegion(runsKernel, -anchor);

  if (bBoundaryOn)
      erode_with_boundary_rle(runsSource, sizeSource.width, sizeSource.height, runsDestination, runsKernel);
  else
      erode_rle(runsSource, runsDestination, runsKernel);
  convertToOutputArray(runsDestination, sizeSource, rlDest);
}


static void subtract_rle( rlVec& regFrom,
                        rlVec& regSubtract,
                        rlVec& regRes)
{
    rlVec::iterator end1 = regFrom.end();
    rlVec::iterator end2 = regSubtract.end();

    rlVec::iterator cur1 = regFrom.begin();
    rlVec::iterator cur2 = regSubtract.begin();
    regRes.clear();

    while( cur1 != end1)
    {
        if (cur2 == end2)
        {
            regRes.insert( regRes.end(), cur1, end1);
            return;
        }
        else if ( cur1->r < cur2->r || (cur1->r == cur2->r && cur1->ce < cur2->cb))
        {
            regRes.push_back(*cur1);
            ++cur1;
        }
        else if ( cur2->r < cur1->r || (cur1->r == cur2->r && cur2->ce < cur1->cb))
            ++cur2;
        else
        {
            int curR = cur1->r;
            assert(curR == cur2->r);
            rlVec::iterator lastIncluded;

            bool bIncremented = false;
            for (lastIncluded = cur2;
                lastIncluded != end2 && lastIncluded->r == curR && lastIncluded->cb <= cur1->ce;
                ++lastIncluded)
            {
                bIncremented = true;
            }

            if (bIncremented)
                --lastIncluded;

            // now all chords from cur2 to lastIncluded have an intersection with cur1
            if (cur1->cb < cur2->cb)
                regRes.push_back(rlType(cur1->cb, cur2->cb - 1, curR));

            // we add the gaps between the chords of reg2 to the result
            while (cur2 < lastIncluded)
            {
                regRes.push_back(rlType(cur2->ce + 1, (cur2 + 1)->cb - 1, curR));
                if (regRes.back().cb > regRes.back().ce)
                {
                    assert(false);
                }
                ++cur2;
            }

            if (cur1->ce > lastIncluded->ce)
            {
                regRes.push_back(rlType(lastIncluded->ce + 1, cur1->ce, curR));
                assert(regRes.back().cb <= regRes.back().ce);
            }
            ++cur1;
        }

    }
}




static void invertRegion(rlVec& runsIn, rlVec& runsOut)
{
    // if there is only one chord in row -> do not insert anything for this row
    // otherwise insert chords for the spaces between chords
    runsOut.clear();
    int nCurRow = std::numeric_limits<int>::min();
    int nLastRight = nCurRow;
    for (rlVec::iterator it = runsIn.begin(); it != runsIn.end(); ++it)
    {
        rlType run = *it;
        if (run.r != nCurRow)
        {
            nCurRow = run.r;
            nLastRight = run.ce;
        }
        else
        {
            assert(run.cb > nLastRight + 1);
            runsOut.push_back(rlType(nLastRight + 1, run.cb - 1, nCurRow));
            nLastRight = run.ce;
        }
    }
}


static void dilate_rle(rlVec& runsSource,
    rlVec& runsDestination,
    rlVec& runsKernel)
{
    cv::Rect rectSource = getBoundingRectangle(runsSource);
    cv::Rect rectKernel = getBoundingRectangle(runsKernel);

    cv::Rect background;
    background.x = rectSource.x - 2 * rectKernel.width;
    background.y = rectSource.y - 2 * rectKernel.height;
    background.width = rectSource.width + 4 * rectKernel.width;
    background.height = rectSource.height + 4 * rectKernel.height;

    rlVec rlBackground, rlSourceInverse, rlResultInverse;
    createUprightRectangle(background, rlBackground);
    subtract_rle(rlBackground, runsSource, rlSourceInverse);

    erode_rle(rlSourceInverse, rlResultInverse, runsKernel);

    invertRegion(rlResultInverse, runsDestination);
}



CV_EXPORTS void dilate(InputArray rlSrc, OutputArray rlDest, InputArray rlKernel, Point anchor)
{
  rlVec runsSource, runsDestination, runsKernel;
  Size sizeSource, sizeKernel;
  convertInputArrayToRuns(rlSrc, runsSource, sizeSource);
  convertInputArrayToRuns(rlKernel, runsKernel, sizeKernel);

  if (anchor != Point(0, 0))
      translateRegion(runsKernel, -anchor);

  dilate_rle(runsSource, runsDestination, runsKernel);

  convertToOutputArray(runsDestination, sizeSource, rlDest);
}

CV_EXPORTS bool isRLMorphologyPossible(InputArray rlStructuringElement)
{
  rlVec runsKernel;
  Size sizeKernel;
  convertInputArrayToRuns(rlStructuringElement, runsKernel, sizeKernel);

  for (int i = 1; i < (int) runsKernel.size(); ++i)
    if (runsKernel[i].r != runsKernel[i-1].r + 1)
      return false;

  return true;
}

CV_EXPORTS void createRLEImage(const std::vector<cv::Point3i>& runs, OutputArray res, Size size)
{
    size_t nRuns = runs.size();
    rlVec runsConverted(nRuns);
    for (size_t i = 0u; i < nRuns; ++i)
    {
        const Point3i &curIn = runs[i];
        runsConverted[i] = rlType(curIn.x, curIn.y, curIn.z);
    }
    sortChords(runsConverted);

    if (size.width == 0 || size.height == 0)
    {
        Rect boundingRect = getBoundingRectangle(runsConverted);
        size.width = std::max(0, boundingRect.x + boundingRect.width);
        size.height = std::max(0, boundingRect.y + boundingRect.height);
    }
    convertToOutputArray(runsConverted, size, res);
}


CV_EXPORTS void morphologyEx(InputArray rlSrc, OutputArray rlDest, int op, InputArray rlKernel,
    bool bBoundaryOnForErosion, Point anchor)
{
    if (op == MORPH_ERODE)
        rl::erode(rlSrc, rlDest, rlKernel, bBoundaryOnForErosion, anchor);
    else if (op == MORPH_DILATE)
        rl::dilate(rlSrc, rlDest, rlKernel, anchor);
    else
    {
        rlVec runsSource, runsKernel, runsDestination;
        Size sizeSource, sizeKernel;
        convertInputArrayToRuns(rlKernel, runsKernel, sizeKernel);
        convertInputArrayToRuns(rlSrc, runsSource, sizeSource);

        if (anchor != Point(0, 0))
            translateRegion(runsKernel, -anchor);

        switch (op)
        {
        case MORPH_OPEN:
        {
            rlVec runsEroded;

            if (bBoundaryOnForErosion)
                erode_with_boundary_rle(runsSource, sizeSource.width, sizeSource.height, runsEroded, runsKernel);
            else
                erode_rle(runsSource, runsEroded, runsKernel);
            dilate_rle(runsEroded, runsDestination, runsKernel);
        }
        break;
        case MORPH_CLOSE:
        {
            rlVec runsDilated;

            dilate_rle(runsSource, runsDilated, runsKernel);
            if (bBoundaryOnForErosion)
                erode_with_boundary_rle(runsDilated, sizeSource.width, sizeSource.height, runsDestination, runsKernel);
            else
                erode_rle(runsDilated, runsDestination, runsKernel);
        }
        break;
        case MORPH_GRADIENT:
        {
            rlVec runsEroded, runsDilated;
            if (bBoundaryOnForErosion)
                erode_with_boundary_rle(runsSource, sizeSource.width, sizeSource.height, runsEroded, runsKernel);
            else
                erode_rle(runsSource, runsEroded, runsKernel);
            dilate_rle(runsSource, runsDilated, runsKernel);
            subtract_rle(runsDilated, runsEroded, runsDestination);
         }
        break;
        case MORPH_TOPHAT:
        {
            rlVec runsEroded, runsOpened;

            if (bBoundaryOnForErosion)
                erode_with_boundary_rle(runsSource, sizeSource.width, sizeSource.height, runsEroded, runsKernel);
            else
                erode_rle(runsSource, runsEroded, runsKernel);
            dilate_rle(runsEroded, runsOpened, runsKernel);
            subtract_rle(runsSource, runsOpened, runsDestination);
        }
        break;

        case MORPH_BLACKHAT:
        {
            rlVec runsClosed, runsDilated;

            dilate_rle(runsSource, runsDilated, runsKernel);
            if (bBoundaryOnForErosion)
                erode_with_boundary_rle(runsDilated, sizeSource.width, sizeSource.height, runsClosed, runsKernel);
            else
                erode_rle(runsDilated, runsClosed, runsKernel);

            subtract_rle(runsClosed, runsSource, runsDestination);
        }
        break;
        default:
        case MORPH_HITMISS:
            CV_Error(Error::StsBadArg, "unknown or unsupported morphological operation");
        }
        convertToOutputArray(runsDestination, sizeSource, rlDest);
    }
}

}
} //end of cv::ximgproc
} //end of cv
