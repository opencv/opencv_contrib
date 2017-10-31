
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/*
Generate Edge Boxes object proposals in given image(s).
Compute Edge Boxes object proposals as described in:
  C. Lawrence Zitnick and Piotr Dollár
  "Edge Boxes: Locating Object Proposals from Edges", ECCV 2014.
The proposal boxes are fast to compute and give state-of-the-art recall.
OpenCV port by: Leonardo Lontra <lhe dot lontra at gmail dot com>
*/

#include "precomp.hpp"

using namespace cv;
using namespace std;

inline int clamp(int v, int min, int max)
{
  return v < min ? min : v > max ? max : v;
}

namespace cv
{
namespace ximgproc
{

class EdgeBoxesImpl : public EdgeBoxes
{
public:

  EdgeBoxesImpl(float alpha,
                  float beta,
                  float eta,
                  float minScore,
                  int maxBoxes,
                  float edgeMinMag,
                  float edgeMergeThr,
                  float clusterMinMag,
                  float maxAspectRatio,
                  float minBoxArea,
                  float gamma,
                  float kappa);

    virtual void getBoundingBoxes(InputArray edge_map, InputArray orientation_map, std::vector<Rect> &boxes);

    float getAlpha() const { return _alpha; }
    void setAlpha(float value)
    {
      _alpha = value;
      _sxStep = sqrt(1 / _alpha);
      _ayStep = (1 + _alpha) / (2 * _alpha);
      _xyStepRatio = (1 - _alpha) / (1 + _alpha);
    }

    float getBeta() const { return _beta; }
    void setBeta(float value) { _beta = value; }

    float getEta() const { return _eta; }
    void setEta(float value) { _eta = value; }

    float getMinScore() const { return _minScore; }
    void setMinScore(float value) { _minScore = value; }

    int getMaxBoxes() const { return _maxBoxes; }
    void setMaxBoxes(int value) { _maxBoxes = value; }

    float getEdgeMinMag() const { return _edgeMinMag; }
    void setEdgeMinMag(float value) { _edgeMinMag = value; }

    float getEdgeMergeThr() const { return _edgeMergeThr; }
    void setEdgeMergeThr(float value) { _edgeMergeThr = value; }

    float getClusterMinMag() const { return _clusterMinMag; }
    void setClusterMinMag(float value) { _clusterMinMag = value; }

    float getMaxAspectRatio() const { return _maxAspectRatio; }
    void setMaxAspectRatio(float value) { _maxAspectRatio = value; }

    float getMinBoxArea() const { return _minBoxArea; }
    void setMinBoxArea(float value) { _minBoxArea = value; }

    float getGamma() const { return _gamma; }
    void setGamma(float value) { _gamma = value; }

    float getKappa() const { return _kappa; }
    void setKappa(float value)
    {
      _kappa = value;
      _scaleNorm.resize(10000);
      for (int i = 0; i < 10000; i++) _scaleNorm[i] = pow(1.f / i, _kappa);
    }

    //! the destructor
    virtual ~EdgeBoxesImpl() {}

private:
    float _alpha;
    float _beta;
    float _eta;
    float _minScore;
    int _maxBoxes;
    float _edgeMinMag;
    float _edgeMergeThr;
    float _clusterMinMag;
    float _maxAspectRatio;
    float _minBoxArea;
    float _gamma;
    float _kappa;

    // edge segment information (see clusterEdges)
    int h, w;                         // image dimensions
    int _segCnt;                      // total segment count
    Mat _segIds;                      // segment ids (-1/0 means no segment)
    vector<float> _segMag;            // segment edge magnitude sums
    vector<Point2i> _segP;            // segment lower-right pixel
    vector<vector<float> > _segAff;   // segment affinities
    vector<vector<int> > _segAffIdx;  // segment neighbors

    // data structures for efficiency (see prepDataStructs)
    Mat _segIImg, _magIImg;
    Mat _hIdxImg, _vIdxImg;
    vector<vector<int> > _hIdxs, _vIdxs;
    vector<float> _scaleNorm;
    float _sxStep, _ayStep, _xyStepRatio;

    // data structures for efficiency (see scoreBox)
    Mat _sWts;
    Mat _sDone, _sMap, _sIds;
    int _sId;

    // helper routines
    static bool boxesCompare(const Box &a, const Box &b) { return a.score < b.score; }
    void clusterEdges(Mat &edgeMap, Mat &orientationMap);
    void prepDataStructs(Mat &edgeMap);
    void scoreAllBoxes(Boxes &boxes);
    void scoreBox(Box &box);
    void refineBox(Box &box);
    float boxesOverlap(Box &a, Box &b);
    void boxesNms(Boxes &boxes, float thr, float eta, int maxBoxes);
};


EdgeBoxesImpl::EdgeBoxesImpl(float alpha,
                             float beta,
                             float eta,
                             float minScore,
                             int maxBoxes,
                             float edgeMinMag,
                             float edgeMergeThr,
                             float clusterMinMag,
                             float maxAspectRatio,
                             float minBoxArea,
                             float gamma,
                             float kappa)
    : _alpha(alpha),
      _beta(beta),
      _eta(eta),
      _minScore(minScore),
      _maxBoxes(maxBoxes),
      _edgeMinMag(edgeMinMag),
      _edgeMergeThr(edgeMergeThr),
      _clusterMinMag(clusterMinMag),
      _maxAspectRatio(maxAspectRatio),
      _minBoxArea(minBoxArea),
      _gamma(gamma),
      _kappa(kappa)

{
  // initialize step sizes
  _sxStep = sqrt(1 / _alpha);
  _ayStep = (1 + _alpha) / (2 * _alpha);
  _xyStepRatio = (1 - _alpha) / (1 + _alpha);

  // create _scaleNorm
  _scaleNorm.resize(10000);
  for (int i = 0; i < 10000; i++) _scaleNorm[i] = pow(1.f / i, _kappa);

}


void EdgeBoxesImpl::clusterEdges(Mat &edgeMap, Mat &orientationMap)
{
    int x, y, xd, yd, i, j;

    // greedily merge connected edge pixels into clusters (create _segIds)
    _segIds = Mat::zeros(w, h, DataType<int>::type);
    _segCnt = 1;
    for (x = 0; x < w; x++)
    {
        const float *e_ptr = edgeMap.ptr<float>(x);
        int *s_ptr = _segIds.ptr<int>(x);
        for (y = 0; y < h; y++)
        {
            if (x == 0 || y == 0 || x == w - 1 || y == h - 1 || e_ptr[y] <= _edgeMinMag)
            {
                s_ptr[y] = -1;
            }
        }
    }

    for (x = 1; x < w - 1; x++)
    {
        int *s_ptr = _segIds.ptr<int>(x);
        for (y = 1; y < h - 1; y++)
        {
            if (s_ptr[y] != 0) continue;
            float sumv = 0;
            int x0 = x;
            int y0 = y;
            vector<float> vs;
            vector<int> xs, ys;

            while (sumv < _edgeMergeThr)
            {
                _segIds.at<int>(x0, y0) = _segCnt;
                float o0 = orientationMap.at<float>(x0, y0);
                float o1, v;
                bool found;
                for (xd = -1; xd <= 1; xd++)
                {
                    const int *s0_ptr = _segIds.ptr<int>(x0 + xd);
                    const float *o_ptr = orientationMap.ptr<float>(x0 + xd);
                    for (yd = -1; yd <= 1; yd++)
                    {
                        if (s0_ptr[y0 + yd] != 0) continue;
                        found = false;
                        for (i = 0; i < (int)xs.size(); i++)
                        {
                            if (xs[i] == x0 + xd && ys[i] == y0 + yd)
                            {
                                found = true;
                                break;
                            }
                        }
                        if (found) continue;
                        o1 = o_ptr[y0 + yd];
                        v = fabs(o1 - o0) / (float)CV_PI;
                        if (v > .5f) v = 1 - v;
                        vs.push_back(v);
                        xs.push_back(x0 + xd);
                        ys.push_back(y0 + yd);
                    }
                }
                float minv = 1000;
                j = 0;
                for (i = 0; i < (int)vs.size(); i++)
                {
                    if (vs[i] < minv)
                    {
                        minv = vs[i];
                        x0 = xs[i];
                        y0 = ys[i];
                        j = i;
                    }
                }
                sumv += minv;
                if (minv < 1000) vs[j] = 1000;
            }

            _segCnt++;
        }
    }

    // merge or remove small segments
    _segMag.resize(_segCnt, 0);
    for (x = 1; x < w - 1; x++)
    {
        const float *e_ptr = edgeMap.ptr<float>(x);
        const int *s_ptr = _segIds.ptr<int>(x);
        for (y = 1; y < h - 1; y++)
        {
          j = s_ptr[y];
          if (j > 0) _segMag[j] += e_ptr[y];
        }
    }

    for (x = 1; x < w - 1; x++)
    {
        int *s_ptr = _segIds.ptr<int>(x);
        for (y = 1; y < h - 1; y++)
        {
          j = s_ptr[y];
          if (j > 0 && _segMag[j] <= _clusterMinMag)
              s_ptr[y] = 0;
        }
    }

    i = 1;
    while (i > 0)
    {
        i = 0;
        for (x = 1; x < w - 1; x++)
        {
            int *s0_ptr = _segIds.ptr<int>(x);
            const float *o0_ptr = orientationMap.ptr<float>(x);
            for (y = 1; y < h - 1; y++)
            {
                if (s0_ptr[y] != 0) continue;
                float o0 = o0_ptr[y];
                float o1, v, minv = 1000;
                j = 0;

                for (xd = -1; xd <= 1; xd++)
                {
                    const int *s1_ptr = _segIds.ptr<int>(x+xd);
                    const float *o1_ptr = orientationMap.ptr<float>(x+xd);
                    for (yd = -1; yd <= 1; yd++)
                    {
                        if (s1_ptr[y + yd] <= 0) continue;
                        o1 = o1_ptr[y + yd];
                        v = fabs(o1 - o0) / (float)CV_PI;
                        if (v > .5f) v = 1 - v;
                        if (v < minv)
                        {
                            minv = v;
                            j = s1_ptr[y + yd];
                        }
                    }
                }

                s0_ptr[y] = j;
                if (j > 0) i++;
            }
        }
    }

    // compactify representation
    _segMag.assign(_segCnt, 0);
    vector<int> map(_segCnt, 0);
    _segCnt = 1;
    for (x = 1; x < w - 1; x++)
    {
        const float *e_ptr = edgeMap.ptr<float>(x);
        const int *s_ptr = _segIds.ptr<int>(x);
        for (y = 1; y < h - 1; y++)
        {
            j = s_ptr[y];
            if (j > 0) _segMag[j] += e_ptr[y];
        }
    }

    for (i = 0; i < (int)_segMag.size(); i++)
    {
      if (_segMag[i] > 0) map[i] = _segCnt++;
    }

    for (x = 1; x < w - 1; x++)
    {
        int *s_ptr = _segIds.ptr<int>(x);
        for (y = 1; y < h - 1; y++)
        {
          j = s_ptr[y];
          if (j > 0) s_ptr[y] = map[j];
        }
    }

    // compute positional means and recompute _segMag
    _segMag.assign(_segCnt, 0);
    vector<float> meanX(_segCnt, 0), meanY(_segCnt, 0);
    vector<float> meanOx(_segCnt, 0), meanOy(_segCnt, 0), meanO(_segCnt, 0);
    for (x = 1; x < w - 1; x++)
    {
        int *s_ptr = _segIds.ptr<int>(x);
        const float *e_ptr = edgeMap.ptr<float>(x);
        const float *o_ptr = orientationMap.ptr<float>(x);
        for (y = 1; y < h - 1; y++)
        {
            j = s_ptr[y];
            if (j <= 0) continue;
            float m = e_ptr[y];
            float o = o_ptr[y];
            _segMag[j] += m;
            meanOx[j] += m * cos(2 * o);
            meanOy[j] += m * sin(2 * o);
            meanX[j] += m * x;
            meanY[j] += m * y;
        }
    }

    for (i = 0; i < _segCnt; i++)
    {
        if (_segMag[i] > 0)
        {
            float m = _segMag[i];
            meanX[i] /= m;
            meanY[i] /= m;
            meanO[i] = atan2(meanOy[i] / m, meanOx[i] / m) / 2;
        }
    }

    // compute segment affinities
    _segAff.resize(_segCnt);
    _segAffIdx.resize(_segCnt);
    for (i = 0; i < _segCnt; i++)
    {
      _segAff[i].resize(0);
      _segAffIdx[i].resize(0);
    }

    const int rad = 2;
    for (x = rad; x < w - rad; x++)
    {
        const int *s0_ptr = _segIds.ptr<int>(x);
        for (y = rad; y < h - rad; y++)
        {
            int s0 = s0_ptr[y];
            if (s0 <= 0) continue;
            for (xd = -rad; xd <= rad; xd++)
            {
                const int *s1_ptr = _segIds.ptr<int>(x+xd);
                for (yd = -rad; yd <= rad; yd++)
                {
                    int s1 = s1_ptr[y + yd];
                    if (s1 <= s0) continue;
                    bool found = false;

                    for (i = 0; i < (int)_segAffIdx[s0].size(); i++)
                    {
                        if (_segAffIdx[s0][i] == s1)
                        {
                            found = true;
                            break;
                        }
                    }
                    if (found) continue;
                    float o = atan2(meanY[s0] - meanY[s1], meanX[s0] - meanX[s1]) + (float)CV_PI / 2.0f;
                    float a = fabs(cos(meanO[s0] - o) * cos(meanO[s1] - o));
                    a = pow(a, _gamma);
                    _segAff[s0].push_back(a);
                    _segAffIdx[s0].push_back(s1);
                    _segAff[s1].push_back(a);
                    _segAffIdx[s1].push_back(s0);
                }
            }
        }
    }

    // compute _segC and _segR
    _segP.resize(_segCnt);
    for (x = 1; x < w - 1; x++)
    {
        const int *s_ptr = _segIds.ptr<int>(x);
        for (y = 1; y < h - 1; y++)
        {
            j = s_ptr[y];
            if (j > 0)
            {
                _segP[j] = Point2i(x, y);
            }
        }
    }
}


void EdgeBoxesImpl::prepDataStructs(Mat &edgeMap)
{
    int y, x, i;

    // create _segIImg
    Mat E1 = Mat::zeros(w, h, DataType<float>::type);

    for (i=0; i < _segCnt; i++)
    {
      if (_segMag[i] > 0) E1.at<float>(_segP[i].x, _segP[i].y) = _segMag[i];
    }

    _segIImg = Mat::zeros(w+1, h+1, DataType<float>::type);
    _magIImg = Mat::zeros(w+1, h+1, DataType<float>::type);

    for (x=1; x < w; x++)
    {
      const float *e_ptr = edgeMap.ptr<float>(x);
      const float *e1_ptr = E1.ptr<float>(x);
      const float *si0_ptr = _segIImg.ptr<float>(x);
      float *si1_ptr = _segIImg.ptr<float>(x+1);
      const float *mi0_ptr = _magIImg.ptr<float>(x);
      float *mi1_ptr =_magIImg.ptr<float>(x+1);
      for (y=1; y < h; y++)
      {
        // create _segIImg
        si1_ptr[y+1] = e1_ptr[y] + si0_ptr[y+1] + si1_ptr[y] - si0_ptr[y];
        float e = e_ptr[y] > _edgeMinMag ? e_ptr[y] : 0;
        // create _magIImg
        mi1_ptr[y+1] = e +mi0_ptr[y+1] + mi1_ptr[y] - mi0_ptr[y];
      }
    }

    // create remaining data structures
    int s = 0;
    int s1;

    _hIdxs.resize(h);
    _hIdxImg = Mat::zeros(w, h, DataType<int>::type);
    for (y = 0; y < h; y++)
    {
        s = 0;
        _hIdxs[y].push_back(s);
        for (x = 0; x < w; x++)
        {
            s1 = _segIds.at<int>(x, y);
            if (s1 != s)
            {
                s = s1;
                _hIdxs[y].push_back(s);
            }
            _hIdxImg.at<int>(x, y) = (int)_hIdxs[y].size() - 1;
        }
    }

    _vIdxs.resize(w);
    _vIdxImg = Mat::zeros(w, h, DataType<int>::type);
    for (x = 0; x < w; x++)
    {
        s = 0;
        _vIdxs[x].push_back(s);
        for (y = 0; y < h; y++)
        {
            s1 = _segIds.at<int>(x, y);
            if (s1 != s)
            {
                s = s1;
                _vIdxs[x].push_back(s);
            }
            _vIdxImg.at<int>(x, y) = (int)_vIdxs[x].size() - 1;
        }
    }

    // initialize scoreBox() data structures
    int n = _segCnt + 1;
    _sWts = Mat::zeros(n, 1, DataType<float>::type);
    _sDone = Mat::zeros(n, 1, DataType<int>::type);
    _sMap = Mat::zeros(n, 1, DataType<int>::type);
    _sIds = Mat::zeros(n, 1, DataType<int>::type);
    for (i = 0; i < n; i++) _sDone.at<int>(0, i) = -1;
    _sId = 0;
}


void EdgeBoxesImpl::scoreBox(Box &box)
{
    int i, j, k, q, bh, bw, y0, x0, y1, x1, y0m, y1m, x0m, x1m;
    float *sWts = (float *)_sWts.data;
    int *sDone = (int *)_sDone.data;
    int *sMap = (int *)_sMap.data;
    int *sIds = (int *)_sIds.data;
    int sId = _sId++;

    // add edge count inside box
    y1 = clamp(box.y + box.h, 0, h - 1);
    y0 = box.y = clamp(box.y, 0, h - 1);
    x1 = clamp(box.x + box.w, 0, w - 1);
    x0 = box.x = clamp(box.x, 0, w - 1);
    bh = box.h = y1 - box.y;
    bh /= 2;
    bw = box.w = x1 - box.x;
    bw /= 2;
    float v = _segIImg.at<float>(x0, y0) + _segIImg.at<float>(x1 + 1, y1 + 1)
              - _segIImg.at<float>(x1 + 1, y0) - _segIImg.at<float>(x0, y1 + 1);

    // subtract middle quarter of edges
    y0m = y0 + bh / 2;
    y1m = y0m + bh;
    x0m = x0 + bw / 2;
    x1m = x0m + bw;
    v -= _magIImg.at<float>(x0m, y0m) + _magIImg.at<float>(x1m + 1, y1m + 1)
         - _magIImg.at<float>(x1m + 1, y0m) - _magIImg.at<float>(x0m, y1m + 1);

    // short circuit computation if impossible to score highly
    float norm = _scaleNorm[bw + bh];
    box.score = v * norm;
    if (box.score < _minScore)
    {
        box.score = 0;
        return;
    }

    // find interesecting segments along four boundaries
    int cs, ce, rs, re, n = 0;
    cs = _hIdxImg.at<int>(x0, y0);
    ce = _hIdxImg.at<int>(x1, y0); // top
    for (i = cs; i <= ce; i++)
    {
        j = _hIdxs[y0][i];
        if (j > 0 && sDone[j] != sId)
        {
            sIds[n] = j;
            sWts[n] = 1;
            sDone[j] = sId;
            sMap[j] = n++;
        }
    }

    cs = _hIdxImg.at<int>(x0, y1);
    ce = _hIdxImg.at<int>(x1, y1); // bottom
    for (i = cs; i <= ce; i++)
    {
        j = _hIdxs[y1][i];
        if (j > 0 && sDone[j] != sId)
        {
            sIds[n] = j;
            sWts[n] = 1;
            sDone[j] = sId;
            sMap[j] = n++;
        }
    }

    rs = _vIdxImg.at<int>(x0, y0);
    re = _vIdxImg.at<int>(x0, y1); // left
    for (i = rs; i <= re; i++)
    {
        j = _vIdxs[x0][i];
        if (j > 0 && sDone[j] != sId)
        {
            sIds[n] = j;
            sWts[n] = 1;
            sDone[j] = sId;
            sMap[j] = n++;
        }
    }

    rs = _vIdxImg.at<int>(x1, y0);
    re = _vIdxImg.at<int>(x1, y1); // right
    for (i = rs; i <= re; i++)
    {
        j = _vIdxs[x1][i];
        if (j > 0 && sDone[j] != sId)
        {
            sIds[n] = j;
            sWts[n] = 1;
            sDone[j] = sId;
            sMap[j] = n++;
        }
    }

    // follow connected paths and set weights accordingly (ws=1 means remove)
    for (i = 0; i < n; i++)
    {
        float ws = sWts[i];
        j = sIds[i];
        for (k = 0; k < (int)_segAffIdx[j].size(); k++)
        {
            q = _segAffIdx[j][k];
            float wq = ws * _segAff[j][k];
            if (wq < .05f) continue; // short circuit for efficiency
            if (sDone[q] == sId)
            {
                if (wq > sWts[sMap[q]])
                {
                    sWts[sMap[q]] = wq;
                    i = min(i, sMap[q] - 1);
                }
            }
            else if (_segP[q].x >= x0 && _segP[q].x <= x1 && _segP[q].y >= y0 && _segP[q].y <= y1)
            {
                sIds[n] = q;
                sWts[n] = wq;
                sDone[q] = sId;
                sMap[q] = n++;
            }
        }
    }
    // finally remove segments connected to boundaries
    for (i = 0; i < n; i++)
    {
        k = sIds[i];
        if (_segP[k].x >= x0 && _segP[k].x <= x1 && _segP[k].y >= y0 && _segP[k].y <= y1) v -= sWts[i] * _segMag[k];
    }

    v *= norm;
    if (v < _minScore) v = 0;
    box.score = v;
}


void EdgeBoxesImpl::refineBox(Box &box)
{
    int yStep = (int)(box.h * _xyStepRatio);
    int xStep = (int)(box.w * _xyStepRatio);
    while (1)
    {
        // prepare for iteration
        yStep /= 2;
        xStep /= 2;
        if (yStep <= 2 && xStep <= 2) break;
        yStep = max(1, yStep);
        xStep = max(1, xStep);
        Box B;
        // search over y start
        B = box;
        B.y = box.y - yStep;
        B.h = B.h + yStep;
        scoreBox(B);

        if (B.score <= box.score)
        {
            B = box;
            B.y = box.y + yStep;
            B.h = B.h - yStep;
            scoreBox(B);
        }
        if (B.score > box.score) box = B;
        // search over y end
        B = box;
        B.h = B.h + yStep;
        scoreBox(B);

        if (B.score <= box.score)
        {
            B = box;
            B.h = B.h - yStep;
            scoreBox(B);
        }
        if (B.score > box.score) box = B;
        // search over x start
        B = box;
        B.x = box.x - xStep;
        B.w = B.w + xStep;
        scoreBox(B);

        if (B.score <= box.score)
        {
            B = box;
            B.x = box.x + xStep;
            B.w = B.w - xStep;
            scoreBox(B);
        }

        if (B.score > box.score) box = B;
        // search over x end
        B = box;
        B.w = B.w + xStep;
        scoreBox(B);

        if (B.score <= box.score)
        {
            B = box;
            B.w = B.w - xStep;
            scoreBox(B);
        }
        if (B.score > box.score) box = B;
    }
}

void EdgeBoxesImpl::scoreAllBoxes(Boxes &boxes)
{
    // get list of all boxes roughly distributed in grid
    boxes.resize(0);
    int ayRad, sxNum;
    float minSize = sqrt(_minBoxArea);
    ayRad = (int)(log(_maxAspectRatio) / log(_ayStep * _ayStep));
    sxNum = (int)(ceil(log(max(w, h) / minSize) / log(_sxStep)));

    for (int s = 0; s < sxNum; s++)
    {
        int a, y, x, bh, bw, ky, kx = -1;
        float ay, sx;
        for (a = 0; a < 2 * ayRad + 1; a++)
        {
            ay = pow(_ayStep, float(a - ayRad));
            sx = minSize * pow(_sxStep, float(s));
            bh = (int)(sx / ay);
            ky = max(2, (int)(bh * _xyStepRatio));
            bw = (int)(sx * ay);
            kx = max(2, (int)(bw * _xyStepRatio));
            for (x = 0; x < w - bw + kx; x += kx)
            {
                for (y = 0; y < h - bh + ky; y += ky)
                {
                    Box b;
                    b.y = y;
                    b.x = x;
                    b.h = bh;
                    b.w = bw;
                    boxes.push_back(b);
                }
            }
        }
    }

    // score all boxes, refine top candidates
    int i, k = 0, m = (int)boxes.size();
    for (i = 0; i < m; i++)
    {
        scoreBox(boxes[i]);
        if (!boxes[i].score) continue;
        k++;
        refineBox(boxes[i]);
    }
    sort(boxes.rbegin(), boxes.rend(), boxesCompare);
    boxes.resize(k);
}


float EdgeBoxesImpl::boxesOverlap(Box &a, Box &b)
{
    float areai, areaj, areaij;
    int y0, y1, x0, x1, y1i, x1i, y1j, x1j;
    y1i = a.y + a.h;
    x1i = a.x + a.w;
    if (a.y >= y1i || a.x >= x1i) return 0;

    y1j = b.y + b.h;
    x1j = b.x + b.w;
    if (a.y >= y1j || a.x >= x1j) return 0;

    areai = (float) a.w * a.h;
    y0 = max(a.y, b.y);
    y1 = min(y1i, y1j);
    areaj = (float) b.w * b.h;
    x0 = max(a.x, b.x);
    x1 = min(x1i, x1j);
    areaij = (float) max(0, y1 - y0) * max(0, x1 - x0);
    return areaij / (areai + areaj - areaij);
}


void EdgeBoxesImpl::boxesNms(Boxes &boxes, float thr, float eta, int maxBoxes)
{
    sort(boxes.rbegin(), boxes.rend(), boxesCompare);
    if (thr > .99f) return;

    const int nBin = 10000;
    const float step = 1 / thr;
    const float lstep = log(step);

    vector<Boxes> kept;
    kept.resize(nBin + 1);
    int n = (int) boxes.size();
    int i = 0;
    int j, k, b;
    int m = 0;
    int d = 1;

    while (i < n && m < maxBoxes)
    {
        b = boxes[i].w * boxes[i].h;

        bool keep = 1;
        b = clamp((int)(ceil(log(float(b)) / lstep)), d, nBin - d);
        for (j = b - d; j <= b + d; j++)
        {
            for (k = 0; k < (int)kept[j].size(); k++)
            {
                if (keep)
                    keep = boxesOverlap(boxes[i], kept[j][k]) <= thr;
            }
        }

        if (keep)
        {
            kept[b].push_back(boxes[i]);
            m++;
        }

        i++;
        if (keep && eta < 1.0f && thr > .5f)
        {
            thr *= eta;
            d = (int)ceil(log(1.0f / thr) / lstep);
        }
    }

    boxes.resize(m);
    i = 0;
    for (j = 0; j < nBin; j++)
    {
        for (k = 0; k < (int)kept[j].size(); k++)
        {
            boxes[i++] = kept[j][k];
        }
    }
    sort(boxes.rbegin(), boxes.rend(), boxesCompare);
}


void EdgeBoxesImpl::getBoundingBoxes(InputArray edge_map, InputArray orientation_map, std::vector<Rect> &boxes)
{
    CV_Assert(edge_map.depth() == CV_32F);
    CV_Assert(orientation_map.depth() == CV_32F);

    Mat E = edge_map.getMat().t();
    Mat O = orientation_map.getMat().t();

    h = E.cols;
    w = E.rows;

    clusterEdges(E, O);
    prepDataStructs(E);

    Boxes b;
    scoreAllBoxes(b);
    boxesNms(b, _beta, _eta, _maxBoxes);

    // create output boxes
    int n = (int) b.size();
    boxes.resize(n);
    for(int i=0; i < n; i++)
    {
        boxes[i] = Rect((int)b[i].x + 1, (int)b[i].y + 1, (int)b[i].w, (int)b[i].h);
    }
}


Ptr<EdgeBoxes> createEdgeBoxes(float alpha,
                              float beta,
                              float eta,
                              float minScore,
                              int   maxBoxes,
                              float edgeMinMag,
                              float edgeMergeThr,
                              float clusterMinMag,
                              float maxAspectRatio,
                              float minBoxArea,
                              float gamma,
                              float kappa)
{
    return makePtr<EdgeBoxesImpl>(alpha,
                                  beta,
                                  eta,
                                  minScore,
                                  maxBoxes,
                                  edgeMinMag,
                                  edgeMergeThr,
                                  clusterMinMag,
                                  maxAspectRatio,
                                  minBoxArea,
                                  gamma,
                                  kappa);
}

}
}
