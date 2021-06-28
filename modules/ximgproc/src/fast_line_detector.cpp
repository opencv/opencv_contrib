// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <vector>
#include <iostream>

struct SEGMENT
{
    float x1, y1, x2, y2, angle;
};

/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////

namespace cv{
namespace ximgproc{

class FastLineDetectorImpl : public FastLineDetector
{
    public:

        FastLineDetectorImpl(int _length_threshold = 10, float _distance_threshold = 1.414213562f,
                double _canny_th1 = 50.0, double _canny_th2 = 50.0, int _canny_aperture_size = 3,
                bool _do_merge = false);

        void detect(InputArray image, OutputArray lines) CV_OVERRIDE;

        void drawSegments(InputOutputArray image, InputArray lines, bool draw_arrow = false, Scalar linecolor = Scalar(0, 0, 255), int linethickness = 1) CV_OVERRIDE;

    private:
        int imagewidth, imageheight, threshold_length;
        float threshold_dist;
        double canny_th1, canny_th2;
        int canny_aperture_size;
        bool do_merge;

        FastLineDetectorImpl& operator= (const FastLineDetectorImpl&); // to quiet MSVC
        template<class T>
            void incidentPoint(const Mat& l, T& pt);

        void mergeLines(const SEGMENT& seg1, const SEGMENT& seg2, SEGMENT& seg_merged);

        bool mergeSegments(const SEGMENT& seg1, const SEGMENT& seg2, SEGMENT& seg_merged);

        bool getPointChain(const Mat& img, Point pt, Point& chained_pt, float& direction, int step);

        double distPointLine(const Mat& p, Mat& l);

        void extractSegments(const std::vector<Point2i>& points, std::vector<SEGMENT>& segments);

        void lineDetection(const Mat& src, std::vector<SEGMENT>& segments_all);

        void pointInboardTest(const Size srcSize, Point2i& pt);

        inline void getAngle(SEGMENT& seg);

        void additionalOperationsOnSegment(const Mat& src, SEGMENT& seg);

        void drawSegment(InputOutputArray image, const SEGMENT& seg, Scalar bgr = Scalar(0,255,0), int thickness = 1, bool directed = true);
};

/////////////////////////////////////////////////////////////////////////////////////////

CV_EXPORTS Ptr<FastLineDetector> createFastLineDetector(
        int length_threshold, float distance_threshold,
        double canny_th1, double canny_th2, int canny_aperture_size, bool do_merge)
{
    return makePtr<FastLineDetectorImpl>(
            length_threshold, distance_threshold,
            canny_th1, canny_th2, canny_aperture_size, do_merge);
}

/////////////////////////////////////////////////////////////////////////////////////////

FastLineDetectorImpl::FastLineDetectorImpl(int _length_threshold, float _distance_threshold,
        double _canny_th1, double _canny_th2, int _canny_aperture_size, bool _do_merge)
    :threshold_length(_length_threshold), threshold_dist(_distance_threshold),
    canny_th1(_canny_th1), canny_th2(_canny_th2), canny_aperture_size(_canny_aperture_size), do_merge(_do_merge)
{
    CV_Assert(_length_threshold > 0 && _distance_threshold > 0 &&
            _canny_th1 > 0 && _canny_th2 > 0 && _canny_aperture_size >= 0);
}

void FastLineDetectorImpl::detect(InputArray _image, OutputArray _lines)
{
    CV_INSTRUMENT_REGION();

    Mat image = _image.getMat();
    CV_Assert(!image.empty() && image.type() == CV_8UC1);

    std::vector<Vec4f> lines;
    std::vector<SEGMENT> segments;
    lineDetection(image, segments);
    for(size_t i = 0; i < segments.size(); ++i)
    {
        const SEGMENT seg = segments[i];
        Vec4f line(seg.x1, seg.y1, seg.x2, seg.y2);
        lines.push_back(line);
    }
    Mat(lines).copyTo(_lines);
}

void FastLineDetectorImpl::drawSegments(InputOutputArray image, InputArray lines, bool draw_arrow, Scalar linecolor, int linethickness)
{
    CV_INSTRUMENT_REGION();

    int cn = image.channels();
    CV_Assert(!image.empty() && ( cn == 1 || cn == 3 || cn == 4));

    if (cn == 1)
    {
        cvtColor(image, image, COLOR_GRAY2BGR);
    }
    else
    {
        cvtColor(image, image, COLOR_BGRA2GRAY);
        cvtColor(image, image, cn == 3 ? COLOR_GRAY2BGR : COLOR_GRAY2BGRA);
    }

    double gap = 10.0;
    double arrow_angle = 30.0;

    Mat _lines;
    _lines = lines.getMat();
    int N = _lines.checkVector(4);
    // Draw segments
    for(int i = 0; i < N; ++i)
    {
        const Vec4f& v = _lines.at<Vec4f>(i);
        Point2f b(v[0], v[1]);
        Point2f e(v[2], v[3]);
        line(image, b, e, linecolor, linethickness);
        if(draw_arrow)
        {
            SEGMENT seg;
            seg.x1 = b.x;
            seg.y1 = b.y;
            seg.x2 = e.x;
            seg.y2 = e.y;
            getAngle(seg);
            double ang = (double)seg.angle;
            Point2i p1;
            p1.x = cvRound(seg.x2 - gap*cos(arrow_angle * CV_PI / 180.0 + ang));
            p1.y = cvRound(seg.y2 - gap*sin(arrow_angle * CV_PI / 180.0 + ang));
            pointInboardTest(image.size(), p1);
            line(image, Point(cvRound(seg.x2), cvRound(seg.y2)), p1, linecolor, linethickness);
        }
    }
}

void FastLineDetectorImpl::mergeLines(const SEGMENT& seg1, const SEGMENT& seg2, SEGMENT& seg_merged)
{
    double xg = 0.0, yg = 0.0;
    double delta1x = 0.0, delta1y = 0.0, delta2x = 0.0, delta2y = 0.0;
    float ax = 0, bx = 0, cx = 0, dx = 0;
    float ay = 0, by = 0, cy = 0, dy = 0;
    double li = 0.0, lj = 0.0;
    double thi = 0.0, thj = 0.0, thr = 0.0;
    double axg = 0.0, bxg = 0.0, cxg = 0.0, dxg = 0.0, delta1xg = 0.0, delta2xg = 0.0;

    ax = seg1.x1;
    ay = seg1.y1;

    bx = seg1.x2;
    by = seg1.y2;
    cx = seg2.x1;
    cy = seg2.y1;

    dx = seg2.x2;
    dy = seg2.y2;

    float dlix = (bx - ax);
    float dliy = (by - ay);
    float dljx = (dx - cx);
    float dljy = (dy - cy);

    li = sqrt((double) (dlix * dlix) + (double) (dliy * dliy));
    lj = sqrt((double) (dljx * dljx) + (double) (dljy * dljy));

    xg = (li * (double) (ax + bx) + lj * (double) (cx + dx))
        / (double) (2.0 * (li + lj));
    yg = (li * (double) (ay + by) + lj * (double) (cy + dy))
        / (double) (2.0 * (li + lj));

    if(dlix == 0.0f) thi = CV_PI / 2.0;
    else thi = atan(dliy / dlix);

    if(dljx == 0.0f) thj = CV_PI / 2.0;
    else thj = atan(dljy / dljx);

    if (fabs(thi - thj) <= CV_PI / 2.0)
    {
        thr = (li * thi + lj * thj) / (li + lj);
    }
    else
    {
        double tmp = thj - CV_PI * (thj / fabs(thj));
        thr = li * thi + lj * tmp;
        thr /= (li + lj);
    }

    axg = ((double) ay - yg) * sin(thr) + ((double) ax - xg) * cos(thr);
    bxg = ((double) by - yg) * sin(thr) + ((double) bx - xg) * cos(thr);
    cxg = ((double) cy - yg) * sin(thr) + ((double) cx - xg) * cos(thr);
    dxg = ((double) dy - yg) * sin(thr) + ((double) dx - xg) * cos(thr);

    delta1xg = min(axg,min(bxg,min(cxg,dxg)));
    delta2xg = max(axg,max(bxg,max(cxg,dxg)));

    delta1x = delta1xg * cos(thr) + xg;
    delta1y = delta1xg * sin(thr) + yg;
    delta2x = delta2xg * cos(thr) + xg;
    delta2y = delta2xg * sin(thr) + yg;

    seg_merged.x1 = (float)delta1x;
    seg_merged.y1 = (float)delta1y;
    seg_merged.x2 = (float)delta2x;
    seg_merged.y2 = (float)delta2y;
}

double FastLineDetectorImpl::distPointLine(const Mat& p, Mat& l)
{
    double x = l.at<double>(0,0);
    double y = l.at<double>(1,0);
    double w = sqrt(x*x+y*y);

    l.at<double>(0,0) = x / w;
    l.at<double>(1,0) = y / w;
    l.at<double>(2,0) = l.at<double>(2,0) / w;

    return l.dot(p);
}

bool FastLineDetectorImpl::mergeSegments(const SEGMENT& seg1, const SEGMENT& seg2, SEGMENT& seg_merged)
{
    double o[] = { 0.0, 0.0, 1.0 };
    double a[] = { 0.0, 0.0, 1.0 };
    double b[] = { 0.0, 0.0, 1.0 };
    double c[3];

    o[0] = ( seg2.x1 + seg2.x2 ) / 2.0;
    o[1] = ( seg2.y1 + seg2.y2 ) / 2.0;

    a[0] = seg1.x1;
    a[1] = seg1.y1;
    b[0] = seg1.x2;
    b[1] = seg1.y2;

    Mat ori = Mat(3, 1, CV_64FC1, o).clone();
    Mat p1 = Mat(3, 1, CV_64FC1, a).clone();
    Mat p2 = Mat(3, 1, CV_64FC1, b).clone();
    Mat l1 = Mat(3, 1, CV_64FC1, c).clone();

    l1 = p1.cross(p2);

    Point2f seg1mid, seg2mid;
    seg1mid.x = (seg1.x1 + seg1.x2) /2.0f;
    seg1mid.y = (seg1.y1 + seg1.y2) /2.0f;
    seg2mid.x = (seg2.x1 + seg2.x2) /2.0f;
    seg2mid.y = (seg2.y1 + seg2.y2) /2.0f;

    float seg1len = sqrt((seg1.x1 - seg1.x2)*(seg1.x1 - seg1.x2)+(seg1.y1 - seg1.y2)*(seg1.y1 - seg1.y2));
    float seg2len = sqrt((seg2.x1 - seg2.x2)*(seg2.x1 - seg2.x2)+(seg2.y1 - seg2.y2)*(seg2.y1 - seg2.y2));
    float middist = sqrt((seg1mid.x - seg2mid.x)*(seg1mid.x - seg2mid.x) + (seg1mid.y - seg2mid.y)*(seg1mid.y - seg2mid.y));
    float angdiff = fabs(seg1.angle - seg2.angle);

    float dist = (float)distPointLine(ori, l1);

    if ( fabs( dist ) <= threshold_dist * 2.0f && middist <= seg1len / 2.0f + seg2len / 2.0f + 20.0f
            && angdiff <= CV_PI / 180.0f * 5.0f)
    {
        mergeLines(seg1, seg2, seg_merged);
        return true;
    }
    else
    {
        return false;
    }
}

template<class T>
    void FastLineDetectorImpl::incidentPoint(const Mat& l, T& pt)
    {
        double a[] = { (double)pt.x, (double)pt.y, 1.0 };
        double b[] = { l.at<double>(0,0), l.at<double>(1,0), 0.0 };
        double c[3];

        Mat xk = Mat(3, 1, CV_64FC1, a).clone();
        Mat lh = Mat(3, 1, CV_64FC1, b).clone();
        Mat lk = Mat(3, 1, CV_64FC1, c).clone();

        lk = xk.cross(lh);
        xk = lk.cross(l);

        xk.convertTo(xk, -1, 1.0 / xk.at<double>(2,0));

        Point2f pt_tmp;
        pt_tmp.x = (float)xk.at<double>(0,0) < 0.0f ? 0.0f : (float)xk.at<double>(0,0)
            >= (imagewidth - 1.0f) ? (imagewidth - 1.0f) : (float)xk.at<double>(0,0);
        pt_tmp.y = (float)xk.at<double>(1,0) < 0.0f ? 0.0f : (float)xk.at<double>(1,0)
            >= (imageheight - 1.0f) ? (imageheight - 1.0f) : (float)xk.at<double>(1,0);
        pt = T(pt_tmp);
    }

void FastLineDetectorImpl::extractSegments(const std::vector<Point2i>& points, std::vector<SEGMENT>& segments)
{
    bool is_line;

    int i, j;
    SEGMENT seg;
    Point2i ps, pe, pt;

    std::vector<Point2i> l_points;

    int total = (int)points.size();

    for ( i = 0; i + threshold_length < total; i++ )
    {
        ps = points[i];
        pe = points[i + threshold_length];

        double a[] = { (double)ps.x, (double)ps.y, 1 };
        double b[] = { (double)pe.x, (double)pe.y, 1 };
        double c[3], d[3];

        Mat p1 = Mat(3, 1, CV_64FC1, a).clone();
        Mat p2 = Mat(3, 1, CV_64FC1, b).clone();
        Mat p = Mat(3, 1, CV_64FC1, c).clone();
        Mat l = Mat(3, 1, CV_64FC1, d).clone();
        l = p1.cross(p2);

        is_line = true;

        l_points.clear();
        l_points.push_back(ps);

        for ( j = 1; j < threshold_length; j++ )
        {
            pt.x = points[i+j].x;
            pt.y = points[i+j].y;

            p.at<double>(0,0) = (double)pt.x;
            p.at<double>(1,0) = (double)pt.y;
            p.at<double>(2,0) = 1.0;

            double dist = distPointLine(p, l);

            if ( fabs( dist ) > threshold_dist )
            {
                is_line = false;
                break;
            }
            l_points.push_back(pt);
        }

        // Line check fail, test next point
        if ( is_line == false )
            continue;

        l_points.push_back(pe);

        Vec4f line;
        fitLine( Mat(l_points), line, DIST_L2, 0, 0.01, 0.01);
        a[0] = line[2];
        a[1] = line[3];
        b[0] = line[2] + line[0];
        b[1] = line[3] + line[1];

        p1 = Mat(3, 1, CV_64FC1, a).clone();
        p2 = Mat(3, 1, CV_64FC1, b).clone();

        l = p1.cross(p2);

        incidentPoint(l, ps);

        // Extending line
        for ( j = threshold_length + 1; i + j < total; j++ )
        {
            pt.x = points[i+j].x;
            pt.y = points[i+j].y;

            p.at<double>(0,0) = (double)pt.x;
            p.at<double>(1,0) = (double)pt.y;
            p.at<double>(2,0) = 1.0;

            double dist = distPointLine(p, l);
            if ( fabs( dist ) > threshold_dist )
            {
                fitLine( Mat(l_points), line, DIST_L2, 0, 0.01, 0.01);
                a[0] = line[2];
                a[1] = line[3];
                b[0] = line[2] + line[0];
                b[1] = line[3] + line[1];

                p1 = Mat(3, 1, CV_64FC1, a).clone();
                p2 = Mat(3, 1, CV_64FC1, b).clone();

                l = p1.cross(p2);
                dist = distPointLine(p, l);
                if ( fabs( dist ) > threshold_dist ) {
                    j--;
                    break;
                }
            }
            pe = pt;
            l_points.push_back(pt);
        }
        fitLine( Mat(l_points), line, DIST_L2, 0, 0.01, 0.01);
        a[0] = line[2];
        a[1] = line[3];
        b[0] = line[2] + line[0];
        b[1] = line[3] + line[1];

        p1 = Mat(3, 1, CV_64FC1, a).clone();
        p2 = Mat(3, 1, CV_64FC1, b).clone();

        l = p1.cross(p2);

        Point2f e1, e2;
        e1.x = (float)ps.x;
        e1.y = (float)ps.y;
        e2.x = (float)pe.x;
        e2.y = (float)pe.y;

        incidentPoint(l, e1);
        incidentPoint(l, e2);
        seg.x1 = e1.x;
        seg.y1 = e1.y;
        seg.x2 = e2.x;
        seg.y2 = e2.y;

        segments.push_back(seg);
        i = i + j;
    }
}

void FastLineDetectorImpl::pointInboardTest(const Size srcSize, Point2i& pt)
{
    pt.x = pt.x <= 5 ? 5 : pt.x >= srcSize.width - 5 ? srcSize.width - 5 : pt.x;
    pt.y = pt.y <= 5 ? 5 : pt.y >= srcSize.height - 5 ? srcSize.height - 5 : pt.y;
}

bool FastLineDetectorImpl::getPointChain(const Mat& img, Point pt,
        Point& chained_pt, float& direction, int step)
{
    int ri, ci;
    int indices[8][2] = { {1,1}, {1,0}, {1,-1}, {0,-1},
        {-1,-1},{-1,0}, {-1,1}, {0,1} };

    float min_dir_diff = 7.0f;
    Point consistent_pt;
    int consistent_direction = 0;
    for ( int i = 0; i < 8; i++ )
    {
        ci = pt.x + indices[i][1];
        ri = pt.y + indices[i][0];

        if ( ri < 0 || ri == img.rows || ci < 0 || ci == img.cols )
            continue;

        if ( img.at<unsigned char>(ri, ci) == 0 )
            continue;

        if(step == 0)
        {
            chained_pt.x = ci;
            chained_pt.y = ri;
            // direction = (float)i;
            direction = i > 4 ? (float)(i - 8) : (float)i;
            return true;
        }
        else
        {
            float curr_dir = i > 4 ? (float)(i - 8) : (float)i;
            float dir_diff = abs(curr_dir - direction);
            dir_diff = dir_diff > 4.0f ? 8.0f - dir_diff : dir_diff;
            if(dir_diff <= min_dir_diff)
            {
                min_dir_diff = dir_diff;
                consistent_pt.x = ci;
                consistent_pt.y = ri;
                consistent_direction = i > 4 ? i - 8 : i;
            }
        }
    }
    if(min_dir_diff < 2.0f)
    {
        chained_pt.x = consistent_pt.x;
        chained_pt.y = consistent_pt.y;
        direction = (direction * (float)step + (float)consistent_direction)
            / (float)(step + 1);
        return true;
    }
    return false;
}

void FastLineDetectorImpl::lineDetection(const Mat& src, std::vector<SEGMENT>& segments_all)
{
    int r, c;
    imageheight=src.rows; imagewidth=src.cols;

    std::vector<Point2i> points;
    std::vector<SEGMENT> segments, segments_tmp;
    Mat canny;
    if (canny_aperture_size == 0)
    {
        canny = src;
    }
    else
    {
        Canny(src, canny, canny_th1, canny_th2, canny_aperture_size);
    }
    canny.colRange(0,6).rowRange(0,6).setTo(cv::Scalar::all(0));
    canny.colRange(src.cols-5,src.cols).rowRange(src.rows-5,src.rows).setTo(cv::Scalar::all(0));

    SEGMENT seg, seg1, seg2;

    for ( r = 0; r < imageheight; r++ )
    {
        for ( c = 0; c < imagewidth; c++ )
        {
            // Find seeds - skip for non-seeds
            if ( canny.at<unsigned char>(r,c) == 0 )
                continue;

            // Found seeds
            Point2i pt = Point2i(c,r);

            points.push_back(pt);
            canny.at<unsigned char>(pt.y, pt.x) = 0;

            float direction = 0.0f;
            int step = 0;
            while(getPointChain(canny, pt, pt, direction, step))
            {
                points.push_back(pt);
                step++;
                canny.at<unsigned char>(pt.y, pt.x) = 0;
            }

            if ( points.size() < (unsigned int)threshold_length + 1 )
            {
                points.clear();
                continue;
            }

            extractSegments(points, segments);

            if ( segments.size() == 0 )
            {
                points.clear();
                continue;
            }
            for ( int i = 0; i < (int)segments.size(); i++ )
            {
                seg = segments[i];
                float length = sqrt((seg.x1 - seg.x2)*(seg.x1 - seg.x2) +
                        (seg.y1 - seg.y2)*(seg.y1 - seg.y2));
                if(length < threshold_length)
                    continue;
                if( (seg.x1 <= 5.0f && seg.x2 <= 5.0f) ||
                    (seg.y1 <= 5.0f && seg.y2 <= 5.0f) ||
                    (seg.x1 >= imagewidth - 5.0f && seg.x2 >= imagewidth - 5.0f) ||
                    (seg.y1 >= imageheight - 5.0f && seg.y2 >= imageheight - 5.0f) )
                    continue;
                additionalOperationsOnSegment(src, seg);
                if(!do_merge)
                    segments_all.push_back(seg);
                segments_tmp.push_back(seg);
            }
            points.clear();
            segments.clear();
        }
    }
    if(!do_merge)
        return;

    bool is_merged = false;
    int ith = (int)segments_tmp.size() - 1;
    int jth = ith - 1;
    while(ith > 1 || jth > 0)
    {
        seg1 = segments_tmp[ith];
        seg2 = segments_tmp[jth];
        SEGMENT seg_merged;
        is_merged = mergeSegments(seg1, seg2, seg_merged);
        if(is_merged == true)
        {
            seg2 = seg_merged;
            additionalOperationsOnSegment(src, seg2);
            std::vector<SEGMENT>::iterator it = segments_tmp.begin() + ith;
            *it = seg2;
            segments_tmp.erase(segments_tmp.begin()+jth);
            ith--;
            jth = ith - 1;
        }
        else
        {
            jth--;
        }
        if(jth < 0) {
            ith--;
            jth = ith - 1;
        }
    }
    segments_all = segments_tmp;
}

inline void FastLineDetectorImpl::getAngle(SEGMENT& seg)
{
    seg.angle = (float)(fastAtan2(seg.y2 - seg.y1, seg.x2 - seg.x1) / 180.0f * CV_PI);
}

void FastLineDetectorImpl::additionalOperationsOnSegment(const Mat& src, SEGMENT& seg)
{
    if(seg.x1 == 0.0f && seg.x2 == 0.0f && seg.y1 == 0.0f && seg.y2 == 0.0f)
        return;

    getAngle(seg);
    double ang = (double)seg.angle;

    Point2f start = Point2f(seg.x1, seg.y1);
    Point2f end = Point2f(seg.x2, seg.y2);

    double dx = 0.0, dy = 0.0;
    dx = (double) end.x - (double) start.x;
    dy = (double) end.y - (double) start.y;

    int num_points = 10;
    Point2f *points = new Point2f[num_points];

    points[0] = start;
    points[num_points - 1] = end;
    for (int i = 0; i < num_points; i++)
    {
        if (i == 0 || i == num_points - 1)
            continue;
        points[i].x = points[0].x + ((float)dx / float(num_points - 1) * (float) i);
        points[i].y = points[0].y + ((float)dy / float(num_points - 1) * (float) i);
    }

    Point2i *points_right = new Point2i[num_points];
    Point2i *points_left = new Point2i[num_points];
    double gap = 1.0;

    for(int i = 0; i < num_points; i++)
    {
        points_right[i].x = cvRound(points[i].x + gap*cos(90.0 * CV_PI / 180.0 + ang));
        points_right[i].y = cvRound(points[i].y + gap*sin(90.0 * CV_PI / 180.0 + ang));
        points_left[i].x = cvRound(points[i].x - gap*cos(90.0 * CV_PI / 180.0 + ang));
        points_left[i].y = cvRound(points[i].y - gap*sin(90.0 * CV_PI / 180.0 + ang));
        pointInboardTest(src.size(), points_right[i]);
        pointInboardTest(src.size(), points_left[i]);
    }

    int iR = 0, iL = 0;
    for(int i = 0; i < num_points; i++)
    {
        iR += src.at<unsigned char>(points_right[i].y, points_right[i].x);
        iL += src.at<unsigned char>(points_left[i].y, points_left[i].x);
    }

    if(iR > iL)
    {
        std::swap(seg.x1, seg.x2);
        std::swap(seg.y1, seg.y2);
        getAngle(seg);
    }

    delete[] points;
    delete[] points_right;
    delete[] points_left;

    return;
}

} // namespace cv
} // namespace ximgproc
