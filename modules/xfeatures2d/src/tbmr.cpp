// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "msd_pyramid.hpp"

namespace cv
{
namespace xfeatures2d
{

class TBMR_Impl CV_FINAL : public TBMR
{
  public:
    struct Params
    {
        Params(int _min_area = 60, float _max_area_relative = 0.01,
               float _scale = 1.5, int _n_scale = -1)
        {
            CV_Assert(_min_area >= 0);
            CV_Assert(_max_area_relative >=
                      std::numeric_limits<float>::epsilon());

            minArea = _min_area;
            maxAreaRelative = _max_area_relative;
            scale = _scale;
            n_scale = _n_scale;
        }

        uint minArea;
        float maxAreaRelative;
        int n_scale;
        float scale;
    };

    explicit TBMR_Impl(const Params &_params) : params(_params) {}

    virtual ~TBMR_Impl() CV_OVERRIDE {}

    virtual void setMinArea(int minArea) CV_OVERRIDE
    {
        params.minArea = std::max(minArea, 0);
    }
    int getMinArea() const CV_OVERRIDE { return params.minArea; }

    virtual void setMaxAreaRelative(float maxAreaRelative) CV_OVERRIDE
    {
        params.maxAreaRelative =
            std::max(maxAreaRelative, std::numeric_limits<float>::epsilon());
    }
    virtual float getMaxAreaRelative() const CV_OVERRIDE
    {
        return params.maxAreaRelative;
    }
    virtual void setScaleFactor(float scale_factor) CV_OVERRIDE
    {
        params.scale = std::max(scale_factor, 1.f);
    }
    virtual float getScaleFactor() const CV_OVERRIDE { return params.scale; }
    virtual void setNScales(int n_scales) CV_OVERRIDE
    {
        params.n_scale = n_scales;
    }
    virtual int getNScales() const CV_OVERRIDE { return params.n_scale; }

    virtual void detect(InputArray image,
                        CV_OUT std::vector<KeyPoint> &keypoints,
                        InputArray mask = noArray()) CV_OVERRIDE;

    virtual void detect(InputArray image,
                        CV_OUT std::vector<Elliptic_KeyPoint> &keypoints,
                        InputArray mask = noArray()) CV_OVERRIDE;

    virtual void
    detectAndCompute(InputArray image, InputArray mask,
                     CV_OUT std::vector<Elliptic_KeyPoint> &keypoints,
                     OutputArray descriptors,
                     bool useProvidedKeypoints = false) CV_OVERRIDE;

    CV_INLINE uint zfindroot(uint *parent, uint p)
    {
        if (parent[p] == p)
            return p;
        else
            return parent[p] = zfindroot(parent, parent[p]);
    }

    // Calculate the Component tree. Based on the order of S, it will be a
    // min or max tree.
    void calcMinMaxTree(Mat ima)
    {
        int rs = ima.rows;
        int cs = ima.cols;
        uint imSize = (uint)rs * cs;

        std::array<int, 4> offsets = {
            -ima.cols, -1, 1, ima.cols
        }; // {-1,0}, {0,-1}, {0,1}, {1,0} yx
        std::array<Vec2i, 4> offsetsv = { Vec2i(0, -1), Vec2i(-1, 0),
                                          Vec2i(1, 0), Vec2i(0, 1) }; //  xy
        AutoBuffer<uint> zparb(imSize);
        AutoBuffer<uint> rootb(imSize);
        AutoBuffer<uint> rankb(imSize);
        memset(rankb.data(), 0, imSize * sizeof(uint));
        uint* zpar = zparb.data();
        uint *root = rootb.data();
        uint *rank = rankb.data();
        parent = Mat(rs, cs, CV_32S); // unsigned
        AutoBuffer<bool> dejaVub(imSize);
        memset(dejaVub.data(), 0, imSize * sizeof(bool));
        bool* dejaVu = dejaVub.data();

        const uint *S_ptr = S.ptr<const uint>();
        uint *parent_ptr = parent.ptr<uint>();
        Vec<uint, 6> *imaAttribute = imaAttributes.ptr<Vec<uint, 6>>();

        for (int i = imSize - 1; i >= 0; --i)
        {
            uint p = S_ptr[i];

            Vec2i idx_p(p % cs, p / cs);
            // make set
            {
                parent_ptr[p] = p;
                zpar[p] = p;
                root[p] = p;
                dejaVu[p] = true;
                imaAttribute[p][0] = 1;                   // area
                imaAttribute[p][1] = idx_p[0];            // sum_x
                imaAttribute[p][2] = idx_p[1];            // sum_y
                imaAttribute[p][3] = idx_p[0] * idx_p[1]; // sum_xy
                imaAttribute[p][4] = idx_p[0] * idx_p[0]; // sum_xx
                imaAttribute[p][5] = idx_p[1] * idx_p[1]; // sum_yy
            }

            uint x = p; // zpar of p
            for (unsigned k = 0; k < offsets.size(); ++k)
            {
                uint q = p + offsets[k];

                Vec2i q_idx = idx_p + offsetsv[k];
                bool inBorder = q_idx[0] >= 0 && q_idx[0] < ima.cols &&
                                q_idx[1] >= 0 &&
                                q_idx[1] < ima.rows; // filter out border cases

                if (inBorder && dejaVu[q]) // remove first check
                                           // obsolete
                {
                    uint r = zfindroot(zpar, q);
                    if (r != x) // make union
                    {
                        parent_ptr[root[r]] = p;
                        // accumulate information
                        imaAttribute[p][0] += imaAttribute[root[r]][0]; // area
                        imaAttribute[p][1] += imaAttribute[root[r]][1]; // sum_x
                        imaAttribute[p][2] += imaAttribute[root[r]][2]; // sum_y
                        imaAttribute[p][3] +=
                            imaAttribute[root[r]][3]; // sum_xy
                        imaAttribute[p][4] +=
                            imaAttribute[root[r]][4]; // sum_xx
                        imaAttribute[p][5] +=
                            imaAttribute[root[r]][5]; // sum_yy

                        if (rank[x] < rank[r])
                        {
                            // we merge p to r
                            zpar[x] = r;
                            root[r] = p;
                            x = r;
                        }
                        else if (rank[r] < rank[p])
                        {
                            // merge r to p
                            zpar[r] = p;
                        }
                        else
                        {
                            // same height
                            zpar[r] = p;
                            rank[p] += 1;
                        }
                    }
                }
            }
        }
    }

    void calculateTBMRs(const Mat &image, std::vector<Elliptic_KeyPoint> &tbmrs,
                        const Mat &mask, float scale, int octave)
    {
        uint imSize = image.cols * image.rows;
        uint maxArea =
            static_cast<uint>(params.maxAreaRelative * imSize * scale);
        uint minArea = static_cast<uint>(params.minArea * scale);

        if (parent.empty() || parent.size != image.size)
            parent = Mat(image.rows, image.cols, CV_32S);

        if (imaAttributes.empty() || imaAttributes.size != image.size)
            imaAttributes = Mat(image.rows, image.cols, CV_32SC(6));

        calcMinMaxTree(image);

        const Vec<uint, 6> *imaAttribute =
            imaAttributes.ptr<const Vec<uint, 6>>();
        const uint8_t *ima_ptr = image.ptr<const uint8_t>();
        const uint *S_ptr = S.ptr<const uint>();
        uint *parent_ptr = parent.ptr<uint>();

        // canonization
        for (uint i = 0; i < imSize; ++i)
        {
            uint p = S_ptr[i];
            uint q = parent_ptr[p];
            if (ima_ptr[parent_ptr[q]] == ima_ptr[q])
                parent_ptr[p] = parent_ptr[q];
        }

        // TBMRs extraction
        //------------------------------------------------------------------------
        // small variant of the given algorithm in the paper. For each
        // critical node having more than one child, we check if the
        // largest region containing this node without any change of
        // topology is above its parent, if not, discard this critical
        // node.
        //
        // note also that we do not select the critical nodes themselves
        // as final TBMRs
        //--------------------------------------------------------------------------

        AutoBuffer<uint> numSonsb(imSize);
        memset(numSonsb.data(), 0, imSize * sizeof(uint));
        uint* numSons = numSonsb.data();
        uint vecNodesSize = imaAttribute[S_ptr[0]][0];               // area
        AutoBuffer<uint> vecNodesb(vecNodesSize);
        memset(vecNodesb.data(), 0, vecNodesSize * sizeof(uint));
        uint *vecNodes = vecNodesb.data(); // area
        uint numNodes = 0;

        // leaf to root propagation to select the canonized nodes
        for (int i = imSize - 1; i >= 0; --i)
        {
            uint p = S_ptr[i];
            if (parent_ptr[p] == p || ima_ptr[p] != ima_ptr[parent_ptr[p]])
            {
                vecNodes[numNodes++] = p;
                if (imaAttribute[p][0] >= minArea) // area
                    numSons[parent_ptr[p]]++;
            }
        }

        AutoBuffer<bool> isSeenb(imSize);
        memset(isSeenb.data(), 0, imSize * sizeof(bool));
        bool *isSeen = isSeenb.data();

        // parent of critical leaf node
        AutoBuffer<bool> isParentofLeafb(imSize);
        memset(isParentofLeafb.data(), 0, imSize * sizeof(bool));
        bool* isParentofLeaf = isParentofLeafb.data();

        for (uint i = 0; i < vecNodesSize; i++)
        {
            uint p = vecNodes[i];
            if (numSons[p] == 0 && numSons[parent_ptr[p]] == 1)
                isParentofLeaf[parent_ptr[p]] = true;
        }

        uint numTbmrs = 0;
        AutoBuffer<uint> vecTbmrsb(numNodes);
        uint* vecTbmrs = vecTbmrsb.data();
        for (uint i = 0; i < vecNodesSize; i++)
        {
            uint p = vecNodes[i];
            if (numSons[p] == 1 && !isSeen[p] && imaAttribute[p][0] <= maxArea)
            {
                uint num_ancestors = 0;
                uint pt = p;
                uint po = pt;
                while (numSons[pt] == 1 && imaAttribute[pt][0] <= maxArea)
                {
                    isSeen[pt] = true;
                    num_ancestors++;
                    po = pt;
                    pt = parent_ptr[pt];
                }
                if (!isParentofLeaf[p] || num_ancestors > 1)
                {
                    vecTbmrs[numTbmrs++] = po;
                }
            }
        }
        // end of TBMRs extraction
        //------------------------------------------------------------------------

        // compute best fitting ellipses
        //------------------------------------------------------------------------
        for (uint i = 0; i < numTbmrs; i++)
        {
            uint p = vecTbmrs[i];
            double area = static_cast<double>(imaAttribute[p][0]);
            double sum_x = static_cast<double>(imaAttribute[p][1]);
            double sum_y = static_cast<double>(imaAttribute[p][2]);
            double sum_xy = static_cast<double>(imaAttribute[p][3]);
            double sum_xx = static_cast<double>(imaAttribute[p][4]);
            double sum_yy = static_cast<double>(imaAttribute[p][5]);

            // Barycenter:
            double x = sum_x / area;
            double y = sum_y / area;

            double i20 = sum_xx - area * x * x;
            double i02 = sum_yy - area * y * y;
            double i11 = sum_xy - area * x * y;
            double n = i20 * i02 - i11 * i11;
            if (n != 0)
            {
                double a = (i02 / n) * (area - 1) / 4;
                double b = (-i11 / n) * (area - 1) / 4;
                double c = (i20 / n) * (area - 1) / 4;

                // filter out some non meaningful ellipses
                double a1 = a;
                double b1 = b;
                double c1 = c;
                uint ai = 0;
                uint bi = 0;
                uint ci = 0;
                if (a > 0)
                {
                    if (a < 0.00005)
                        a1 = 0;
                    else if (a < 0.0001)
                    {
                        a1 = 0.0001;
                    }
                    else
                    {
                        ai = (uint)(10000 * a);
                        a1 = (double)ai / 10000;
                    }
                }
                else
                {
                    if (a > -0.00005)
                        a1 = 0;
                    else if (a > -0.0001)
                        a1 = -0.0001;
                    else
                    {
                        ai = (uint)(10000 * (-a));
                        a1 = -(double)ai / 10000;
                    }
                }

                if (b > 0)
                {
                    if (b < 0.00005)
                        b1 = 0;
                    else if (b < 0.0001)
                    {
                        b1 = 0.0001;
                    }
                    else
                    {
                        bi = (uint)(10000 * b);
                        b1 = (double)bi / 10000;
                    }
                }
                else
                {
                    if (b > -0.00005)
                        b1 = 0;
                    else if (b > -0.0001)
                        b1 = -0.0001;
                    else
                    {
                        bi = (uint)(10000 * (-b));
                        b1 = -(double)bi / 10000;
                    }
                }

                if (c > 0)
                {
                    if (c < 0.00005)
                        c1 = 0;
                    else if (c < 0.0001)
                    {
                        c1 = 0.0001;
                    }
                    else
                    {
                        ci = (uint)(10000 * c);
                        c1 = (double)ci / 10000;
                    }
                }
                else
                {
                    if (c > -0.00005)
                        c1 = 0;
                    else if (c > -0.0001)
                        c1 = -0.0001;
                    else
                    {
                        ci = (uint)(10000 * (-c));
                        c1 = -(double)ci / 10000;
                    }
                }
                double v =
                    (a1 + c1 -
                     std::sqrt(a1 * a1 + c1 * c1 + 4 * b1 * b1 - 2 * a1 * c1)) /
                    2;

                double l1 = 1. / std::sqrt((a + c +
                                            std::sqrt(a * a + c * c +
                                                      4 * b * b - 2 * a * c)) /
                                           2);
                double l2 = 1. / std::sqrt((a + c -
                                            std::sqrt(a * a + c * c +
                                                      4 * b * b - 2 * a * c)) /
                                           2);
                double minAxL = std::min(l1, l2);
                double majAxL = std::max(l1, l2);

                if (minAxL >= 1.5 && v != 0 &&
                    (mask.empty() ||
                     mask.at<uchar>(cvRound(y), cvRound(x)) != 0))
                {
                    double theta = 0;
                    if (b == 0)
                        if (a < c)
                            theta = 0;
                        else
                            theta = CV_PI / 2.;
                    else
                        theta = CV_PI / 2. + 0.5 * std::atan2(2 * b, (a - c));

                    float size = (float)majAxL;

                    // not sure if we should scale or not scale x,y,axes,size
                    // (as scale is stored in si)
                    Elliptic_KeyPoint ekp(
                        Point2f((float)x, (float)y) * scale, (float)theta,
                        cv::Size2f((float)majAxL, (float)minAxL) * scale,
                        size * scale, scale);
                    ekp.octave = octave;
                    tbmrs.push_back(ekp);
                }
            }
        }
        //---------------------------------------------
    }

    Mat tempsrc;

    // component tree representation (parent,S): see
    // https://ieeexplore.ieee.org/document/6850018
    Mat parent;
    Mat S;
    // moments: compound type of: (area, x, y, xy, xx, yy)
    Mat imaAttributes;

    Params params;
};

void TBMR_Impl::detect(InputArray _image, std::vector<KeyPoint> &keypoints,
                       InputArray _mask)
{
    std::vector<Elliptic_KeyPoint> kp;
    detect(_image, kp, _mask);
    keypoints.resize(kp.size());
    for (size_t i = 0; i < kp.size(); ++i)
        keypoints[i] = kp[i];
}

void TBMR_Impl::detect(InputArray _image,
                       std::vector<Elliptic_KeyPoint> &keypoints,
                       InputArray _mask)
{
    Mat mask = _mask.getMat();
    Mat src = _image.getMat();

    keypoints.clear();

    if (src.empty())
        return;

    if (!mask.empty())
    {
        CV_Assert(mask.type() == CV_8UC1);
        CV_Assert(mask.size == src.size);
    }

    if (!src.isContinuous())
    {
        src.copyTo(tempsrc);
        src = tempsrc;
    }

    CV_Assert(src.depth() == CV_8U);

    if (src.channels() != 1)
        cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);

    int m_cur_n_scales =
        params.n_scale > 0
            ? params.n_scale
            : 1 /*todo calculate optimal scale factor from image size*/;
    float m_scale_factor = params.scale;

    // track and eliminate duplicates introduced with multi scale position ->
    // (size)
    Mat dupl(src.rows / 4, src.cols / 4, CV_32F, cv::Scalar::all(0));
    float *dupl_ptr = dupl.ptr<float>();

    std::vector<Mat> pyr;
    MSDImagePyramid scaleSpacer(src, m_cur_n_scales, m_scale_factor);
    pyr = scaleSpacer.getImPyr();

    int oct = 0;
    for (auto &s : pyr)
    {
        float scale = ((float)s.cols) / pyr.begin()->cols;
        std::vector<Elliptic_KeyPoint> kpts;

        // append max tree tbmrs
        sortIdx(s.reshape(1, 1), S,
                SortFlags::SORT_ASCENDING | SortFlags::SORT_EVERY_ROW);
        calculateTBMRs(s, kpts, mask, scale, oct);

        // reverse instead of sort
        flip(S, S, -1);
        calculateTBMRs(s, kpts, mask, scale, oct);

        if (oct == 0)
        {
            for (const auto &k : kpts)
            {
                dupl_ptr[(int)(k.pt.x / 4) +
                         (int)(k.pt.y / 4) * (src.cols / 4)] = k.size;
            }
            keypoints.insert(keypoints.end(), kpts.begin(), kpts.end());
        }
        else
        {
            for (const auto &k : kpts)
            {
                float &sz = dupl_ptr[(int)(k.pt.x / 4) +
                                     (int)(k.pt.y / 4) * (src.cols / 4)];
                // we hereby add only features that are at least 4 pixels away
                // or have a significantly different size
                if (std::abs(k.size - sz) / std::max(k.size, sz) >= 0.2f)
                {
                    sz = k.size;
                    keypoints.push_back(k);
                }
            }
        }

        oct++;
    }
}

void TBMR_Impl::detectAndCompute(
    InputArray image, InputArray mask,
    CV_OUT std::vector<Elliptic_KeyPoint> &keypoints, OutputArray descriptors,
    bool useProvidedKeypoints)
{
    // We can use SIFT to compute descriptors for the extracted keypoints...
    auto sift = SIFT::create();
    auto dac = AffineFeature2D::create(this, sift);
    dac->detectAndCompute(image, mask, keypoints, descriptors,
                          useProvidedKeypoints);
}

Ptr<TBMR> TBMR::create(int _min_area, float _max_area_relative, float _scale,
                       int _n_scale)
{
    return cv::makePtr<TBMR_Impl>(
        TBMR_Impl::Params(_min_area, _max_area_relative, _scale, _n_scale));
}

} // namespace xfeatures2d
} // namespace cv