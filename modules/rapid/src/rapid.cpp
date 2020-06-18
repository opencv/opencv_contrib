// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv
{
namespace rapid
{

static std::vector<int> getSilhoutteVertices(const Size& imsize, const std::vector<Point>& contour,
                                             const Mat_<Point2f>& pts2d)
{
    // store indices
    Mat_<int> img1(imsize, 0);
    Rect img_rect({0, 0}, imsize);
    for (int i = 0; i < pts2d.rows; i++) {
        if (img_rect.contains(pts2d(i))) {
            img1(pts2d(i)) = i + 1;
        }
    }

    std::vector<int> v_idx;
    // look up indices on contour
    for (size_t i = 0; i < contour.size(); i++) {
        if (int idx = img1(contour[i])) {
            v_idx.push_back(idx - 1);
        }
    }

    return v_idx;
}

class Contour3DSampler {
    std::vector<int> idx;        // indices of points on contour
    std::vector<float> cum_dist; // prefix sum

    Mat_<Point2f> ipts2d;
    Mat_<Point3f> ipts3d;

    float lambda;
    int pos;

public:
    float perimeter;

    Contour3DSampler(const Mat_<Point2f>& pts2d, const Mat_<Point3f>& pts3d,
                     const std::vector<Point>& contour, const Size& imsize)
        : ipts2d(pts2d), ipts3d(pts3d)
    {
        idx = getSilhoutteVertices(imsize, contour, pts2d);

        CV_Assert(!idx.empty());
        // close the loop
        idx.push_back(idx[0]);

        cum_dist.resize(idx.size());
        perimeter = 0.0f;

        for (size_t i = 1; i < idx.size(); i++) {
            perimeter += (float)norm(pts2d(idx[i]) - pts2d(idx[i - 1]));
            cum_dist[i] = perimeter;
        }

        pos = 0;
        lambda = 0;
    }

    void advanceTo(float dist)
    {
        while (pos < int(cum_dist.size() - 1) && dist >= cum_dist[pos]) {
            pos++;
        }

        lambda = (dist - cum_dist[pos - 1]) / (cum_dist[pos] - cum_dist[pos - 1]);
    }

    Point3f current3D() const { return (1 - lambda) * ipts3d(idx[pos - 1]) + lambda * ipts3d(idx[pos]); }
    Point2f current2D() const { return (1 - lambda) * ipts2d(idx[pos - 1]) + lambda * ipts2d(idx[pos]); }
};

void drawWireframe(InputOutputArray img, InputArray _pts2d, InputArray _tris,
                   const Scalar& color, int type, bool cullBackface)
{
    CV_Assert(_tris.getMat().checkVector(3, CV_32S) > 0);
    CV_Assert(_pts2d.getMat().checkVector(2, CV_32F) > 0);

    Mat_<Vec3i> tris = _tris.getMat();
    Mat_<Point2f> pts2d = _pts2d.getMat();

    for (int i = 0; i < int(tris.total()); i++) {
        const auto& idx = tris(i);
        std::vector<Point> poly = {pts2d(idx[0]), pts2d(idx[1]), pts2d(idx[2])};

        // skip back facing triangles
        if (cullBackface && ((poly[2] - poly[0]).cross(poly[2] - poly[1]) >= 0))
            continue;

        polylines(img, poly, true, color, 1, type);
    }
}

void drawSearchLines(InputOutputArray img, InputArray _locations, const Scalar& color)
{
    Mat locations = _locations.getMat();
    CV_CheckTypeEQ(_locations.type(), CV_16SC2, "Vec2s data type expected");

    for (int i = 0; i < locations.rows; i++) {
        Point pt1(locations.at<Vec2s>(i, 0));
        Point pt2(locations.at<Vec2s>(i, locations.cols - 1));
        line(img, pt1, pt2, color, 1);
    }
}

static void sampleControlPoints(int num, Contour3DSampler& sampler, const Rect& roi, OutputArray _opts2d,
                                OutputArray _opts3d)
{
    std::vector<Vec3f> opts3d;
    opts3d.reserve(num);
    std::vector<Vec2f> opts2d;
    opts2d.reserve(num);

    // sample at equal steps
    float step = sampler.perimeter / num;

    if (step == 0)
        num = 0; // edge case -> skip loop

    for (int i = 0; i < num; i++) {
        sampler.advanceTo(step * i);
        auto pt2d = sampler.current2D();

        // skip points too close to border
        if (!roi.contains(pt2d))
            continue;

        opts3d.push_back(sampler.current3D());
        opts2d.push_back(pt2d);
    }

    Mat(opts3d).copyTo(_opts3d);
    Mat(opts2d).copyTo(_opts2d);
}

void extractControlPoints(int num, int len, InputArray pts3d, InputArray rvec, InputArray tvec,
                          InputArray K, const Size& imsize, InputArray tris, OutputArray ctl2d,
                          OutputArray ctl3d)
{
    CV_Assert(num);

    Mat_<Point2f> pts2d(pts3d.rows(), 1);
    projectPoints(pts3d, rvec, tvec, K, noArray(), pts2d);

    Mat_<uchar> img(imsize, uchar(0));
    drawWireframe(img, pts2d, tris.getMat(), 255, LINE_8, true);

    // find contour
    std::vector<std::vector<Point>> contours;
    findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    CV_Assert(!contours.empty());

    Contour3DSampler sampler(pts2d, pts3d.getMat(), contours[0], imsize);
    Rect valid_roi(Point(len, len), imsize - Size(2 * len, 2 * len));
    sampleControlPoints(num, sampler, valid_roi, ctl2d, ctl3d);
}

void extractLineBundle(int len, InputArray ctl2d, InputArray img, OutputArray bundle,
                       OutputArray srcLocations)
{
    CV_Assert(len > 0);
    Mat _img = img.getMat();

    CV_Assert(ctl2d.getMat().checkVector(2, CV_32F) > 0);
    Mat_<Point2f> contour = ctl2d.getMat();

    const int N = contour.rows;
    const int W = len * 2 + 1;

    srcLocations.create(N, W, CV_16SC2);
    Mat_<Vec2s> _srcLocations = srcLocations.getMat();

    for (int i = 0; i < N; i++) {
        // central difference
        const Point2f diff = contour((i + 1) % N) - contour((i - 1 + N) % N);
        Point2f n(normalize(Vec2f(-diff.y, diff.x))); // perpendicular to diff
        // make it cover L pixels
        n *= len / std::max(std::abs(n.x), std::abs(n.y));

        LineIterator li(_img, contour(i) - n, contour(i) + n);
        CV_DbgAssert(li.count == W);

        for (int j = 0; j < li.count; j++, ++li) {
            _srcLocations(i, j) = Vec2i(li.pos());
        }
    }

    remap(img, bundle, srcLocations, noArray(),
          INTER_NEAREST); // inter_nearest as we use integer locations
}

void compute1DSobel(const Mat& src, Mat& dst)
{
    CV_CheckDepthEQ(src.depth(), CV_8U, "only uchar images supported");
    int channels = src.channels();

    CV_Assert(channels == 1 || channels == 3);

    dst.create(src.size(), CV_8U);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            // central difference kernel: [-1, 0, 1]
            if (channels == 3) {
                const Vec3s diff = Vec3s(src.at<Vec3b>(i, j + 1)) - Vec3s(src.at<Vec3b>(i, j - 1));
                dst.at<uchar>(i, j) =
                    (uchar)std::max(std::max(std::abs(diff[0]), std::abs(diff[1])), std::abs(diff[2]));
            } else {
                dst.at<uchar>(i, j) = (uchar)std::abs(src.at<uchar>(i, j + 1) - src.at<uchar>(i, j - 1));
            }
        }
        dst.at<uchar>(i, 0) = dst.at<uchar>(i, src.cols - 1) = 0; // border
    }
}

void findCorrespondencies(InputArray bundle, OutputArray _cols, OutputArray _response)
{
    Mat_<uchar> sobel;
    compute1DSobel(bundle.getMat(), sobel);

    _cols.create(sobel.rows, 1, CV_32S);
    Mat_<int> cols = _cols.getMat();

    Mat_<uchar> response;
    if (_response.needed()) {
        _response.create(sobel.rows, 1, CV_8U);
        response = _response.getMat();
    }

    // sobel.cols = 2*len + 1
    const int len = sobel.cols / 2;
    const int ct = len + 1;

    // find closest maximum to center
    for (int i = 0; i < sobel.rows; i++) {
        int pos = ct;
        uchar mx = sobel.at<uchar>(i, ct);
        for (int j = 0; j < len; j++) {
            uchar right = sobel.at<uchar>(i, ct + j);
            uchar left = sobel.at<uchar>(i, ct - j);
            if (right > mx) {
                mx = right;
                pos = ct + j;
            }
            if (left > mx) {
                mx = left;
                pos = ct - j;
            }
        }

        if (!response.empty())
            response(i) = mx;

        cols(i) = pos;
    }
}

void drawCorrespondencies(InputOutputArray _bundle, InputArray _cols, InputArray _colors)
{
    CV_CheckTypeEQ(_cols.type(), CV_32S, "cols must be of int type");
    CV_Assert(_bundle.rows() == _cols.rows());
    CV_Assert(_colors.empty() || _colors.rows() == _cols.rows());

    Mat bundle = _bundle.getMat();
    Mat_<int> cols = _cols.getMat();
    Mat_<Vec4d> colors = _colors.getMat();

    for (int i = 0; i < bundle.rows; i++) {
        bundle(Rect(Point(cols(i), i), Size(1, 1))) = colors.empty() ? Scalar::all(255) : colors(i);
    }
}

void convertCorrespondencies(InputArray _cols, InputArray _srcLocations, OutputArray _pts2d,
                             InputOutputArray _pts3d, InputArray _mask)
{
    CV_CheckTypeEQ(_cols.type(), CV_32S, "cols must be of int type");
    CV_CheckTypeEQ(_srcLocations.type(), CV_16SC2, "Vec2s data type expected");
    CV_Assert(_srcLocations.rows() == _cols.rows());

    Mat_<cv::Vec2s> srcLocations = _srcLocations.getMat();
    Mat_<int> cols = _cols.getMat();

    Mat pts2d = Mat(0, 1, CV_16SC2);
    pts2d.reserve(cols.rows);

    Mat_<uchar> mask;
    if (!_mask.empty())
    {
        CV_CheckTypeEQ(_mask.type(), CV_8UC1, "mask must be of uchar type");
        CV_Assert(_cols.rows() == _mask.rows());
        mask = _mask.getMat();
    }

    Mat pts3d;
    Mat opts3d;
    if(!_pts3d.empty())
    {
        CV_Assert(_cols.rows() == _pts3d.rows());
        pts3d = _pts3d.getMat();
        opts3d.create(0, 1, pts3d.type());
        opts3d.reserve(cols.rows);
    }

    for (int i = 0; i < cols.rows; i++) {
        if (!mask.empty() && !mask(i))
            continue;

        pts2d.push_back(srcLocations(i, cols(i)));
        if(!pts3d.empty())
            opts3d.push_back(pts3d.row(i));
    }

    pts2d.copyTo(_pts2d);
    if(!pts3d.empty())
        opts3d.copyTo(_pts3d);
}

float rapid(InputArray img, int num, int len, InputArray vtx, InputArray tris, InputArray K,
            InputOutputArray rvec, InputOutputArray tvec, double* rmsd)
{
    CV_Assert(num >= 3);
    Mat pts2d, pts3d;
    extractControlPoints(num, len, vtx, rvec, tvec, K, img.size(), tris, pts2d, pts3d);
    if (pts2d.empty())
        return 0;

    Mat lineBundle, imgLoc;
    extractLineBundle(len, pts2d, img, lineBundle, imgLoc);

    Mat cols, response;
    findCorrespondencies(lineBundle, cols, response);

    const uchar sobel_thresh = 20;
    Mat mask = response > sobel_thresh;
    convertCorrespondencies(cols, imgLoc, pts2d, pts3d, mask);

    if(rmsd)
    {
        cols.copyTo(cols, mask);
        cols -= len + 1;
        *rmsd = std::sqrt(norm(cols, NORM_L2SQR) / cols.rows);
    }

    if (pts2d.rows < 3)
        return 0;

    solvePnPRefineLM(pts3d, pts2d, K, cv::noArray(), rvec, tvec);

    return float(pts2d.rows) / num;
}

Tracker::~Tracker() {}

struct RapidImpl : public Rapid
{
    Mat pts3d;
    Mat tris;
    RapidImpl(InputArray _pts3d, InputArray _tris)
    {
        CV_Assert(_tris.getMat().checkVector(3, CV_32S) > 0);
        CV_Assert(_pts3d.getMat().checkVector(3, CV_32F) > 0);
        pts3d = _pts3d.getMat();
        tris = _tris.getMat();
    }
    float compute(InputArray img, int num, int len, InputArray K, InputOutputArray rvec,
                  InputOutputArray tvec, const TermCriteria& termcrit) CV_OVERRIDE
    {
        float ret = 0;
        int niter = std::max(1, termcrit.maxCount);

        double rmsd;
        Mat cols;
        for(int i = 0; i < niter; i++)
        {
            ret = rapid(img, num, len, pts3d, tris, K, rvec, tvec,
                        termcrit.type & TermCriteria::EPS ? &rmsd : NULL);

            if((termcrit.type & TermCriteria::EPS) && rmsd < termcrit.epsilon)
            {
                break;
            }
        }
        return ret;
    }

    void clearState() CV_OVERRIDE
    {
        // nothing to do
    }
};

Ptr<Rapid> Rapid::create(InputArray pts3d, InputArray tris)
{
    return makePtr<RapidImpl>(pts3d, tris);
}

} /* namespace rapid */
} /* namespace cv */
