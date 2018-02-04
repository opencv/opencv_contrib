// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "hfs_core.hpp"
#include <vector>
using namespace std;

namespace cv{ namespace hfs{


const Point DIRECTION4[5] =
{
    Point(-1, 0),
    Point(0, -1),
    Point(1, 0),
    Point(0, 1),
    Point(0, 0)
};

const Point CIRCLE2[13] =
{
    Point(0, 1), Point(0, 2), Point(1, 1),
    Point(1, 0), Point(2, 0), Point(1, -1),
    Point(0, -1), Point(0, -2), Point(-1, -1),
    Point(-1, 0), Point(-2, 0), Point(-1, 1),
    Point(0, 0)
};

HfsCore::HfsCore(int height, int width,
    float segThresholdI, int minRegionSizeI, float segThresholdII, int minRegionSizeII,
    float spatialWeight, int spixelSize, int numIter)
{
    hfsSettings.egbThresholdI = segThresholdI;
    hfsSettings.minRegionSizeI = minRegionSizeI;
    hfsSettings.egbThresholdII = segThresholdII;
    hfsSettings.minRegionSizeII = minRegionSizeII;

    hfsSettings.slicSettings.img_size.y = height;
    hfsSettings.slicSettings.img_size.x = width;
    hfsSettings.slicSettings.coh_weight = spatialWeight;
    hfsSettings.slicSettings.spixel_size = spixelSize;
    hfsSettings.slicSettings.num_iters = numIter;
    constructEngine();

    float weight1[] = { -0.0024710407f, 0.00608298f,
        0.0047505307f, 0.0051097558f, 0.00089799752f };
    float weight2[] = { -0.0040629096f, 0.010430338f,
        0.0092625152f, 0.004976281f, 0.0037279273f };
    w1.resize(sizeof(weight1) / sizeof(weight1[0]));
    w2.resize(sizeof(weight2) / sizeof(weight2[0]));
    memcpy(w1.data(), weight1, sizeof(weight1));
    memcpy(w2.data(), weight2, sizeof(weight2));
}

void HfsCore::constructEngine()
{
    mag_engine = Ptr<Magnitude>(
        new Magnitude(hfsSettings.slicSettings.img_size.y,
            hfsSettings.slicSettings.img_size.x));
#ifdef _HFS_CUDA_ON_
    gslic_engine = Ptr<slic::engines::CoreEngine>(
        new slic::engines::CoreEngine(hfsSettings.slicSettings));
    in_img = Ptr<UChar4Image>(
        new UChar4Image(hfsSettings.slicSettings.img_size));
    out_img = Ptr<UChar4Image>(
        new UChar4Image(hfsSettings.slicSettings.img_size));
#endif
}

void HfsCore::reconstructEngine()
{
#ifdef _HFS_CUDA_ON_
    gslic_engine = Ptr<slic::engines::CoreEngine>(
        new slic::engines::CoreEngine(hfsSettings.slicSettings));
#endif
}

HfsCore::~HfsCore(){}

void HfsCore::loadImage( const Mat& inimg, Ptr<UChar4Image> outimg )
{
    Vector4u* outimg_ptr = outimg->getCpuData();
    for ( int y = 0; y < inimg.rows; y++ )
    {
        const Vec3b *ptr = inimg.ptr<Vec3b>(y);
        for ( int x = 0; x < inimg.cols; x++ )
        {
            int idx = x + y * inimg.cols;
            outimg_ptr[idx].z = ptr[x][0];
            outimg_ptr[idx].y = ptr[x][1];
            outimg_ptr[idx].x = ptr[x][2];
        }
    }
}



Mat HfsCore::getSLICIdxCpu(const Mat& img3u, int &num_css)
{
    const int _h = img3u.rows;
    const int _w = img3u.cols;
    const int _s = _h*_w;

    slic::cSLIC cslic;
    vector<int> idx_img = cslic.generate_superpixels(img3u,
        hfsSettings.slicSettings.spixel_size, hfsSettings.slicSettings.coh_weight);

    num_css = 0;
    int _max =
        (int)ceil((float)_w / 8.0f)*(int)ceil((float)_h / 8.0f);
    vector<int> indexes(_max, 0);
    for (int i = 0; i < _s; i++)
        indexes[idx_img[i]]++;
    for (int i = 0; i < _max; i++)
        indexes[i] = (indexes[i] != 0) ? num_css++ : 0;
    for (int i = 0; i < _s; i++)
        idx_img[i] = indexes[idx_img[i]];
    Mat idx_mat(_h, _w, CV_32S, idx_img.data());
    idx_mat.convertTo(idx_mat, CV_16U);
    return idx_mat;
}

Vec4f HfsCore::getColorFeature( const Vec3f& in1, const Vec3f& in2 )
{
    Vec4f feature;
    Vec3f diff = (in1 - in2);
    feature[0] = abs(diff[0]);
    feature[1] = abs(diff[1]);
    feature[2] = abs(diff[2]);
    feature[3] = getEulerDistance( in1, in2 );
    return feature;
}

int HfsCore::getAvgGradientBdry( const Mat& idx_mat,
    const vector<Mat>& mag1us, int num_css, Mat& bd_num,
    vector<Mat>& gradients )
{
    const int _h = idx_mat.rows;
    const int _w = idx_mat.cols;
    const int size = (int)mag1us.size();

    gradients.resize(size);
    for (int i = 0; i < size; i++)
    {
        gradients[i].create(num_css, num_css, CV_32F);
        gradients[i] = Scalar::all(0);
    }
    bd_num.create(num_css, num_css, CV_32F);
    bd_num = Scalar::all(0);

    for (int r = 1; r < _h - 1; r++)
    for (int c = 1; c < _w - 1; c++)
    {
        ushort curr = idx_mat.at<ushort>(r, c);
        ushort pre, tmp = 0, v[4];
        Point p1(c, r), p2;
        for (int k = 0; k < 4; k++)
        {
            p2 = p1 + DIRECTION4[k];
            pre = idx_mat.at<ushort>(p2);
            if (pre != curr)
            {
                bool flag = true;
                for (int t = 0; t < tmp; t++)
                {
                    if (v[t] == pre)
                        flag = false;
                }
                if (flag)
                    v[tmp++] = pre;
            }
        }

        if (tmp > 0)
        {
            for (int n = 0; n < size; n++)
            {
                int u[13]; float m[13];
                for (int k = 0; k < 13; k++)
                {
                    p2 = p1 + CIRCLE2[k];
                    if (!(p2.x >= 0 && p2.x < _w && p2.y >= 0 && p2.y < _h))
                    {
                        u[k] = -1, m[k] = 0;
                        continue;
                    }
                    u[k] = idx_mat.at<ushort>(p2);
                    m[k] = mag1us[n].at<uchar>(p2);
                }

                for (int t = 0; t < tmp; t++)
                {
                    float m_max = 0;
                    for (int k = 0; k < 13; k++)
                    {
                        if ((u[k] == curr || u[k] == v[t]) && m[k] > m_max)
                            m_max = m[k];
                    }
                    gradients[n].at<float>(curr, v[t]) += m_max;
                    gradients[n].at<float>(v[t], curr) += m_max;
                    bd_num.at<float>(curr, v[t])++;
                    bd_num.at<float>(v[t], curr)++;
                }
            }
        }
    }

    int num = 0;
    for (int r_ = 0; r_ < num_css; r_++)
    for (int c_ = 0; c_ < num_css; c_++)
    {
        if (abs(bd_num.at<float>(r_, c_)) > DOUBLE_EPS)
        {
            for (int i = 0; i < size; i++)
                gradients[i].at<float>(r_, c_) /= bd_num.at<float>(r_, c_);
            num++;
        }
    }
    return num;
}

void HfsCore::getSegmentationI( const Mat &lab3u, const Mat &mag1u,
    const Mat &idx_mat, float c, int min_size, Mat &seg, int &num_css)
{
    const int _h = lab3u.rows;
    const int _w = lab3u.cols;

    vector<vector<int> > adjacent(num_css), bdPixNum(num_css);
    vector<vector<float> > bdGradient(num_css);
    for (int r_ = 1; r_ < _h - 1; r_++)
    for (int c_ = 1; c_ < _w - 1; c_++)
    {
        ushort curr = idx_mat.at<ushort>(r_, c_);
        for (int k = 0; k < 4; k++)
        {
            Point p = Point(c_, r_) + DIRECTION4[k];
            ushort pre = idx_mat.at<ushort>(p);
            if (curr > pre)
            {
                float maxG = max(mag1u.at<uchar>(p), mag1u.at<uchar>(r_, c_));
                vector<int>::iterator iter =
                    find(adjacent[curr].begin(), adjacent[curr].end(), pre);
                if (iter == adjacent[curr].end())
                {
                    adjacent[curr].push_back(pre);
                    bdGradient[curr].push_back(maxG);
                    bdPixNum[curr].push_back(1);
                }
                else
                {
                    int temp = (int)(iter - adjacent[curr].begin());
                    bdGradient[curr][temp] += maxG;
                    bdPixNum[curr][temp] += 1;
                }
            }
        }
    }
    for (size_t i = 0; i < (size_t)num_css; i++)
    for (size_t j = 0; j < adjacent[i].size(); j++)
        bdGradient[i][j] /= bdPixNum[i][j];

    int num = 0;
    for (int i = 0; i < num_css; i++)
        num += (int)adjacent[i].size();

    vector<int> numR(num_css, 0);
    vector<Vec3f> avg_color(num_css, Vec3f(0, 0, 0));
    for (int r_ = 0; r_ < _h; r_++)
    {
        const ushort *iP = idx_mat.ptr<ushort>(r_);
        const Vec3b *cP = lab3u.ptr<Vec3b>(r_);
        for (int c_ = 0; c_ < _w; c_++)
            avg_color[iP[c_]] += cP[c_], numR[iP[c_]]++;
    }
    for (int i = 0; i < num_css; i++)
        avg_color[i] /= numR[i];

    vector<Edge> edges(num);
    int index = 0;
    for (int i = 0; i < num_css; i++)
    {
        int adjaNum = (int)adjacent[i].size();
        for (int j = 0; j < adjaNum; j++)
        {
            edges[index].a = i;
            edges[index].b = adjacent[i][j];
            Vec4f fcolor =
                getColorFeature(avg_color[i], avg_color[adjacent[i][j]]);

            edges[index++].w =
                fcolor[0] * w1[0] + fcolor[1] * w1[1]+
                fcolor[2] * w1[2] + fcolor[3] * w1[3]+
                bdGradient[i][j] * w1[4];
        }
    }
    CV_Assert(num == index);

    Ptr<RegionSet> regions = egb_merge(num_css, num, edges, c, numR);
    for (int i = 0; i < num; i++)
    {
        int a = regions->find(edges[i].a);
        int b = regions->find(edges[i].b);
        if ((a != b) && ((regions->numPix(a) < min_size) || (regions->numPix(b) < min_size)))
            regions->join(a, b);
    }

    int idx = 1; vector<int> reg2ind(num_css), indexes(num_css);
    std::memset(indexes.data(), 0, num_css*sizeof(int));
    for (int i = 0; i < num_css; i++)
    {
        int comp = regions->find(i);
        if (!indexes[comp])
            indexes[comp] = idx++;
        reg2ind[i] = indexes[comp];
    }
    CV_Assert(regions->num_sets() == idx - 1);
    seg.create(_h, _w, CV_16U);
    for (int r_ = 0; r_ < _h; r_++)
    {
        ushort *sP = seg.ptr<ushort>(r_);
        const ushort *iP = idx_mat.ptr<ushort>(r_);
        for (int c_ = 0; c_ < _w; c_++)
            sP[c_] = (ushort)reg2ind[iP[c_]];
    }
    num_css = idx;
}

void HfsCore::getSegmentationII(
    const Mat &lab3u, const Mat &mag1u, const Mat &idx_mat,
    float c, int min_size, Mat &seg, int &num_css)
{
    const int _h = lab3u.rows;
    const int _w = lab3u.cols;

    vector<Mat> mag1us, gradients;
    Mat bd_num, texture;
    mag1us.push_back(mag1u);
    int num = getAvgGradientBdry(idx_mat, mag1us,
        num_css, bd_num, gradients);
    // const int size = (int)gradients.size();
    CV_Assert(num % 2 == 0);
    num /= 2;

    vector<int> num_pix(num_css, 0);
    vector<Vec3f> avg_color(num_css, Vec3f(0, 0, 0));
    for (int r_ = 0; r_ < _h; r_++)
    {
        const ushort *idx_ptr = idx_mat.ptr<ushort>(r_);
        const Vec3b *clr_ptr = lab3u.ptr<Vec3b>(r_);
        for (int c_ = 0; c_ < _w; c_++)
            num_pix[idx_ptr[c_]]++, avg_color[idx_ptr[c_]] += clr_ptr[c_];
    }
    for (int i = 1; i < num_css; i++)
        avg_color[i] /= num_pix[i];

    vector<Edge> edges(num);
    int index = 0;
    for (int r_ = 0; r_ < num_css; r_++)
    for (int c_ = 0; c_ < r_; c_++)
    {
        if (bd_num.at<int>(r_, c_) == 0) continue;
        edges[index].a = r_;
        edges[index].b = c_;
        Vec4f fcolor = getColorFeature(avg_color[r_], avg_color[c_]);
        edges[index].w =
            fcolor[0] * w2[0] + fcolor[1] * w2[1]+
            fcolor[2] * w2[2] + fcolor[3] * w2[3];
        edges[index].w += gradients[0].at<float>(r_, c_)*w2[4];
        index++;
    }
    CV_Assert(num == index);

    Ptr<RegionSet> regions = egb_merge(num_css, num, edges, c, num_pix);
    for (int i = 0; i < num; i++)
    {
        int a = regions->find(edges[i].a);
        int b = regions->find(edges[i].b);
        if ((a != b) && ((regions->numPix(a) < min_size)
            || (regions->numPix(b) < min_size)))
            regions->join(a, b);
    }

    int idx = 1;
    vector<int> reg2ind(num_css), indexes(num_css, 0);
    for (int i = 1; i < num_css; i++)
    {
        int comp = regions->find(i);
        if (!indexes[comp])
            indexes[comp] = idx++;
        reg2ind[i] = indexes[comp];
    }
    CV_Assert(regions->num_sets() == idx);
    seg.create(_h, _w, CV_16U);
    for (int r_ = 0; r_ < _h; r_++)
    {
        ushort *sP = seg.ptr<ushort>(r_);
        const ushort *iP = idx_mat.ptr<ushort>(r_);
        for (int c_ = 0; c_ < _w; c_++)
            sP[c_] = (ushort)reg2ind[iP[c_]];
    }

    num_css = idx - 1;
}

void HfsCore::drawSegmentationRes( const Mat& seg,
    const Mat& img3u, int num_css, Mat &show )
{
    const int _h = img3u.rows;
    const int _w = img3u.cols;

    vector<int> region_size(num_css, 0);
    vector<Vec3f> avg_color(num_css, Vec3f(0, 0, 0));
    for (int r = 0; r < _h; r++)
    {
        const Vec3b* imP = img3u.ptr<Vec3b>(r);
        const ushort* segP = seg.ptr<ushort>(r);
        for (int c = 0; c < _w; c++)
        {
            avg_color[segP[c] - 1] += imP[c];
            region_size[segP[c] - 1]++;
        }
    }
    for (int i = 0; i < num_css; i++)
        avg_color[i] /= region_size[i];

    show.create(img3u.size(), img3u.type());
    for (int r = 0; r < _h; r++)
    {
        Vec3b *data = show.ptr<Vec3b>(r);
        const ushort* seg_ptr = seg.ptr<ushort>(r);
        for (int c = 0; c < _w; c++)
            data[c] = avg_color[seg_ptr[c] - 1];
    }
}

int HfsCore::processImageCpu(const Mat &img3u, Mat &seg)
{
    Mat idx_mat, lab3u, mag1u, tmp;
    int num_css;

    idx_mat = getSLICIdxCpu(img3u, num_css);
    cv::cvtColor(img3u, lab3u, COLOR_BGR2Lab);

    mag_engine->processImgCpu(img3u, mag1u);

    getSegmentationI( lab3u, mag1u, idx_mat,
        hfsSettings.egbThresholdI, hfsSettings.minRegionSizeI, tmp, num_css );
    getSegmentationII(lab3u, mag1u, tmp,
        hfsSettings.egbThresholdII, hfsSettings.minRegionSizeII, seg, num_css);
    return num_css;
}

int HfsCore::processImageGpu(const Mat &img3u, Mat &seg)
{
#ifdef _HFS_CUDA_ON_
    Mat idx_mat, lab3u, mag1u, tmp;
    int num_css;

    idx_mat = getSLICIdxGpu(img3u, num_css);
    cv::cvtColor(img3u, lab3u, COLOR_BGR2Lab);

    mag_engine->processImgGpu(img3u, mag1u);

    getSegmentationI(lab3u, mag1u, idx_mat,
        hfsSettings.egbThresholdI, hfsSettings.minRegionSizeI, tmp, num_css);
    getSegmentationII(lab3u, mag1u, tmp,
        hfsSettings.egbThresholdII, hfsSettings.minRegionSizeII, seg, num_css);
    return num_css;
#else
    return processImageCpu(img3u, seg);
#endif
}

#ifdef _HFS_CUDA_ON_
Mat HfsCore::getSLICIdxGpu(const Mat& img3u, int &num_css)
{
    const int _h = img3u.rows;
    const int _w = img3u.cols;
    const int _s = _h*_w;

    loadImage(img3u, in_img);
    gslic_engine->setImageSize(img3u.cols, img3u.rows);

    gslic_engine->processFrame(in_img);
    const IntImage *idx_img = gslic_engine->getSegRes();
    int* idx_img_ptr = (int*)idx_img->getCpuData();

    num_css = 0;
    int _max =
        (int)ceil((float)_w / 8.0f)*(int)ceil((float)_h / 8.0f);
    vector<int> indexes(_max, 0);
    for (int i = 0; i < _s; i++)
        indexes[idx_img_ptr[i]]++;
    for (int i = 0; i < _max; i++)
        indexes[i] = (indexes[i] != 0) ? num_css++ : 0;
    for (int i = 0; i < _s; i++)
        idx_img_ptr[i] = indexes[idx_img_ptr[i]];
    Mat idx_mat(_h, _w, CV_32S, idx_img_ptr);
    idx_mat.convertTo(idx_mat, CV_16U);
    return idx_mat;
}

#endif

}}
