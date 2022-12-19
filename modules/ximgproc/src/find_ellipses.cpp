// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <opencv2/core.hpp>
#include <unordered_map>
#include <numeric>

namespace cv {
namespace ximgproc {

// typedef
typedef std::vector<Point> VP;
typedef std::vector<VP> VVP;
typedef unsigned int uint;

// ellipse format
struct Ellipse {
    Point2f center;
    float a, b;
    float radius;
    float score;

    Ellipse() {
        center = Point2f(0, 0);
        a = 0.f, b = 0.f;
        radius = 0.f, score = 0.f;
    };

    Ellipse(Point2f _center, float _a, float _b, float _radius,
            float _score = 0.f) {
        center = _center;
        this->a = _a, this->b = _b;
        this->radius = _radius, this->score = _score;
    };

    Ellipse(float _center_x, float _center_y, float _a, float _b, float _radius,
            float _score = 0.f) {
        center = Point2f(_center_x, _center_y);
        this->a = _a, this->b = _b;
        this->radius = _radius, this->score = _score;
    };

    Ellipse(const Ellipse &other) {
        center = other.center;
        a = other.a, b = other.b;
        radius = other.radius, score = other.score;
    };

    bool operator<(const Ellipse &other) const {
        if (score == other.score) {
            float lhs_e = b / a;
            float rhs_e = other.b / other.a;
            if (lhs_e == rhs_e)
                return false;
            return lhs_e > rhs_e;
        }
        return score > other.score;
    };

    virtual ~Ellipse() = default;
};

static int inline signal(float val) {
    return val > 0.f ? 1 : -1;
}

static bool sortPoint(const Point &lhs, const Point &rhs) {
    if (lhs.x == rhs.x) {
        return lhs.y < rhs.y;
    }
    return lhs.x < rhs.x;
}

static Point2f lineCrossPoint(Point2f l1p1, Point2f l1p2, Point2f l2p1, Point2f l2p2) {
    Point2f crossPoint;
    float k1, k2, b1, b2;
    if (l1p1.x == l1p2.x && l2p1.x == l2p2.x) {
        crossPoint = Point2f(0, 0);
        return crossPoint;
    }
    if (l1p1.x == l1p2.x) {
        crossPoint.x = l1p1.x;
        k2 = (l2p2.y - l2p1.y) / (l2p2.x - l2p1.x);
        b2 = l2p1.y - k2 * l2p1.x;
        crossPoint.y = k2 * crossPoint.x + b2;
        return crossPoint;
    }
    if (l2p1.x == l2p2.x) {
        crossPoint.x = l2p1.x;
        k2 = (l1p2.y - l1p1.y) / (l1p2.x - l1p1.x);
        b2 = l1p1.y - k2 * l1p1.x;
        crossPoint.y = k2 * crossPoint.x + b2;
        return crossPoint;
    }

    k1 = (l1p2.y - l1p1.y) / (l1p2.x - l1p1.x);
    k2 = (l2p2.y - l2p1.y) / (l2p2.x - l2p1.x);
    b1 = l1p1.y - k1 * l1p1.x;
    b2 = l2p1.y - k2 * l2p1.x;
    if (k1 == k2)
        crossPoint = Point2f(0, 0);
    else {
        crossPoint.x = (b2 - b1) / (k1 - k2);
        crossPoint.y = k1 * crossPoint.x + b1;
    }
    return crossPoint;
}

static void pointToMat(Point2f p1, Point2f p2, Mat& mat) {
    mat.at<float>(0, 0) = p1.x;
    mat.at<float>(0, 1) = p1.y;
    mat.at<float>(1, 0) = p2.x;
    mat.at<float>(1, 1) = p2.y;
}

static float valueOfPoints(Point2f p3, Point2f p2, Point2f p1, Point2f p4, Point2f p5, Point2f p6) {
    float result = 1;

    Point2f v = lineCrossPoint(p1, p2, p3, p4);
    Point2f w = lineCrossPoint(p5, p6, p3, p4);
    Point2f u = lineCrossPoint(p5, p6, p1, p2);

    Mat B = Mat(2, 2, CV_32F);
    Mat C = Mat(2, 2, CV_32F);

    pointToMat(u, v, B);
    pointToMat(p1, p2, C);
    Mat A = C * B.inv();
    result *=
            A.at<float>(0, 0) * A.at<float>(1, 0) / (A.at<float>(0, 1) * A.at<float>(1, 1));

    pointToMat(p3, p4, C);
    pointToMat(v, w, B);
    A = C * B.inv();
    result *=
            A.at<float>(0, 0) * A.at<float>(1, 0) / (A.at<float>(0, 1) * A.at<float>(1, 1));

    pointToMat(p5, p6, C);
    pointToMat(w, u, B);
    A = C * B.inv();
    result *=
            A.at<float>(0, 0) * A.at<float>(1, 0) / (A.at<float>(0, 1) * A.at<float>(1, 1));
    return result;
}

static float inline pointDistance2(const Point &A, const Point &B) {
    return float(((B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y)));
}

static float getMinAnglePI(float alpha, float beta) {
    auto pi = float(CV_PI);
    auto pi2 = float(2.0 * CV_PI);

    // normalize data in [0, 2*pi]
    float a = fmod(alpha + pi2, pi2);
    float b = fmod(beta + pi2, pi2);

    // normalize data in [0, pi]
    if (a > pi)
        a -= pi;
    if (b > pi)
        b -= pi;

    if (a > b)
        swap(a, b);

    float diff1 = b - a;
    float diff2 = pi - diff1;

    return min(diff1, diff2);
}

// ellipse data
struct EllipseData {
    bool isValid;
    float ta, tb;
    float ra, rb;
    Point2f Ma, Mb, Cab;
    std::vector<float> Sa, Sb;
};

// implement of ellipse detector
class EllipseDetectorImpl {

    // preprocessing - Gaussian Smooth.
    Size _kernelSize; // size of the Gaussian filter
    double _sigma; // sigma of the Gaussian filter

    // selection strategy - Step 1 - Discard noisy or straight arcs.
    int _minEdgeLength; // minimum edge size
    float _minOrientedRectSide; // minimum size of the oriented bounding box containing the arc

    // selection strategy - Step 2 - Remove according to mutual convexity.
    float _positionThreshold;

    // selection strategy - Step 3 - Number of points considered for slope estimation when estimating the center.
    unsigned _uNs; // find at most Ns parallel chords

    // selection strategy - Step 3 - Discard pairs of arcs if their estimated center is not close enough
    float _maxCenterDistance; // maximum distance in pixel between 2 center points
    float _maxCenterDistance2; // pow(_maxCenterDistance, 2)

    // validation - Points within a threshold are considered to lie on the ellipse contour.
    float _maxDistanceToEllipseContour;  // maximum distance between a point and the contour

    // validation - Assign a score.
    float _minScore; // minimum score to confirm a detection
    float _minReliability; // minimum auxiliary score to confirm a detection

    // auxiliary variables
    Size _imgSize; // input image size

    int ACC_N_SIZE, ACC_R_SIZE, ACC_A_SIZE; // size of accumulator
    int *accN, *accR, *accA; // pointer to accumulator

public:
    float countsOfFindEllipse;
    float countsOfGetFastCenter;

    EllipseDetectorImpl();

    ~EllipseDetectorImpl() = default;

    // detect the ellipses in the gray image
    void detect(Mat1b &image, std::vector<Ellipse> &ellipses);

    // set the parameters of the detector
    void setParameters(float maxCenterDistance, float minScore, float minReliability);

private:
    // keys for hash table
    static const ushort PAIR_12 = 0x00;
    static const ushort PAIR_23 = 0x01;
    static const ushort PAIR_34 = 0x02;
    static const ushort PAIR_14 = 0x03;

    static uint inline generateKey(uchar pair, ushort u, ushort v);

    void preProcessing(Mat1b &image, Mat1b &dp, Mat1b &dn);

    void clusterEllipses(std::vector<Ellipse> &ellipses);

    static float
    getMedianSlope(std::vector<Point2f> &med, Point2f &centers, std::vector<float> &slopes);

    void getFastCenter(std::vector<Point> &e1, std::vector<Point> &e2, EllipseData &data);

    void detectEdges13(Mat1b &DP, VVP &points_1, VVP &points_3);

    void detectEdges24(Mat1b &DN, VVP &points_2, VVP &points_4);

    void
    findEllipses(Point2f &center, VP &edge_i, VP &edge_j, VP &edge_k, EllipseData &data_ij,
                 EllipseData &data_ik, std::vector<Ellipse> &ellipses);

    static Point2f getCenterCoordinates(EllipseData &data_ij, EllipseData &data_ik);

    void
    getTriplets124(VVP &pi, VVP &pj, VVP &pk, std::unordered_map<uint, EllipseData> &data,
                   std::vector<Ellipse> &ellipses);

    void
    getTriplets231(VVP &pi, VVP &pj, VVP &pk, std::unordered_map<uint, EllipseData> &data,
                   std::vector<Ellipse> &ellipses);

    void
    getTriplets342(VVP &pi, VVP &pj, VVP &pk, std::unordered_map<uint, EllipseData> &data,
                   std::vector<Ellipse> &ellipses);

    void
    getTriplets413(VVP &pi, VVP &pj, VVP &pk, std::unordered_map<uint, EllipseData> &data,
                   std::vector<Ellipse> &ellipses);

    static void labeling(Mat1b &image, VVP &segments, int minLength);
};

EllipseDetectorImpl::EllipseDetectorImpl() {
    // Default Parameters Settings
    _kernelSize = Size(5, 5);
    _sigma = 1.0;
    _positionThreshold = 1.0f;
    _maxCenterDistance = 100.0f * 0.05f;
    _maxCenterDistance2 = _maxCenterDistance * _maxCenterDistance;
    _minEdgeLength = 16;
    _minOrientedRectSide = 3.0f;
    _maxDistanceToEllipseContour = 0.1f;
    _minScore = 0.7f;
    _minReliability = 0.5;
    _uNs = 16;
}

void EllipseDetectorImpl::setParameters(float maxCenterDistance, float minScore,
                                        float minReliability) {
    _maxCenterDistance = maxCenterDistance;
    _minScore = minScore;
    _minReliability = minReliability;

    _maxCenterDistance2 = _maxCenterDistance * _maxCenterDistance;
}

uint inline EllipseDetectorImpl::generateKey(uchar pair, ushort u, ushort v) {
    return (pair << 30) + (u << 15) + v;
}

float EllipseDetectorImpl::getMedianSlope(std::vector<Point2f> &med, Point2f &centers,
                                          std::vector<float> &slopes) {
    // med		: vector of points
    // centers  : centroid of the points in med
    // slopes	: vector of the slopes

    size_t pointCount = med.size();
    // CV_Assert(pointCount >= 2);

    size_t halfSize = pointCount >> 1;
    size_t quarterSize = halfSize >> 1;

    std::vector<float> xx, yy;
    slopes.reserve(halfSize);
    xx.reserve(pointCount);
    yy.reserve(pointCount);

    for (unsigned i = 0; i < halfSize; i++) {
        Point2f &p1 = med[i];
        Point2f &p2 = med[halfSize + i];

        xx.push_back(p1.x);
        xx.push_back(p2.x);
        yy.push_back(p1.y);
        yy.push_back(p2.y);

        float den = (p2.x - p1.x);
        float num = (p2.y - p1.y);

        den = (std::fabs(den) >= 1e-5) ? den : 0.00001f;  // FIXIT: algorithm is not reliable

        slopes.push_back(num / den);
    }

    nth_element(slopes.begin(), slopes.begin() + quarterSize, slopes.end());
    nth_element(xx.begin(), xx.begin() + halfSize, xx.end());
    nth_element(yy.begin(), yy.begin() + halfSize, yy.end());
    centers.x = xx[halfSize];
    centers.y = yy[halfSize];

    return slopes[quarterSize];
}

void EllipseDetectorImpl::getFastCenter(std::vector<Point> &e1, std::vector<Point> &e2,
                                        EllipseData &data) {
    countsOfGetFastCenter++;
    data.isValid = true;

    auto size_1 = unsigned(e1.size());
    auto size_2 = unsigned(e2.size());

    unsigned hsize_1 = size_1 >> 1;
    unsigned hsize_2 = size_2 >> 1;

    Point &med1 = e1[hsize_1];
    Point &med2 = e2[hsize_2];

    Point2f M12, M34;
    float q2, q4;

    {
        auto dx_ref = float(e1[0].x - med2.x);
        auto dy_ref = float(e1[0].y - med2.y);

        if (dx_ref == 0)
            dx_ref = 0.00001f;

        float m_ref = dy_ref / dx_ref;
        data.ra = m_ref;

        // find points with same slope as reference
        std::vector<Point2f> med;
        med.reserve(hsize_2);

        unsigned minPoints = (_uNs < hsize_2) ? _uNs : hsize_2;

        std::vector<uint> indexes(minPoints);
        if (_uNs < hsize_2) {
            unsigned iSzBin = hsize_2 / unsigned(_uNs);
            unsigned iIdx = hsize_2 + (iSzBin / 2);

            for (unsigned i = 0; i < _uNs; i++) {
                indexes[i] = iIdx;
                iIdx += iSzBin;
            }
        } else
            iota(indexes.begin(), indexes.end(), hsize_2);

        for (uint ii = 0; ii < minPoints; ii++) {
            uint i = indexes[ii];

            auto x1 = float(e2[i].x);
            auto y1 = float(e2[i].y);

            uint begin = 0;
            uint end = size_1 - 1;

            auto xb = float(e1[begin].x);
            auto yb = float(e1[begin].y);
            float res_begin = ((xb - x1) * dy_ref) - ((yb - y1) * dx_ref);
            int sign_begin = signal(res_begin);
            if (sign_begin == 0) {
                med.emplace_back((xb + x1) * 0.5f, (yb + y1) * 0.5f);
                continue;
            }

            auto xe = float(e1[end].x);
            auto ye = float(e1[end].y);
            float res_end = ((xe - x1) * dy_ref) - ((ye - y1) * dx_ref);
            int sign_end = signal(res_end);
            if (sign_end == 0) {
                med.emplace_back((xe + x1) * 0.5f, (ye + y1) * 0.5f);
                continue;
            }

            if ((sign_begin + sign_end) != 0)
                continue;


            // search parallel arc
            uint j = (begin + end) >> 1;
            while (end - begin > 2) {
                auto x2 = float(e1[j].x);
                auto y2 = float(e1[j].y);
                float res = ((x2 - x1) * dy_ref) - ((y2 - y1) * dx_ref);
                int sign_res = signal(res);

                if (sign_res == 0) {
                    med.emplace_back((x2 + x1) * 0.5f, (y2 + y1) * 0.5f);
                    break;
                }

                if (sign_res + sign_begin == 0) {
                    sign_end = sign_res;
                    end = j;
                } else {
                    sign_begin = sign_res;
                    begin = j;
                }
                j = (begin + end) >> 1;
            }

            med.emplace_back((e1[j].x + x1) * 0.5f, (e1[j].y + y1) * 0.5f);
        }

        if (med.size() < 2) {
            data.isValid = false;
            return;
        }

        q2 = getMedianSlope(med, M12, data.Sa);
    }

    {
        auto dx_ref = float(med1.x - e2[0].x);
        auto dy_ref = float(med1.y - e2[0].y);

        if (dx_ref == 0)
            dx_ref = 0.00001f;

        float m_ref = dy_ref / dx_ref;
        data.rb = m_ref;

        // find points with same slope as reference
        std::vector<Point2f> med;
        med.reserve(hsize_1);

        uint minPoints = (_uNs < hsize_1) ? _uNs : hsize_1;

        std::vector<uint> indexes(minPoints);
        if (_uNs < hsize_1) {
            unsigned iSzBin = hsize_1 / unsigned(_uNs);
            unsigned iIdx = hsize_1 + (iSzBin / 2);

            for (unsigned i = 0; i < _uNs; i++) {
                indexes[i] = iIdx;
                iIdx += iSzBin;
            }
        } else
            iota(indexes.begin(), indexes.end(), hsize_1);


        for (uint ii = 0; ii < minPoints; ii++) {
            uint i = indexes[ii];

            auto x1 = float(e1[i].x);
            auto y1 = float(e1[i].y);

            uint begin = 0;
            uint end = size_2 - 1;

            auto xb = float(e2[begin].x);
            auto yb = float(e2[begin].y);
            float res_begin = ((xb - x1) * dy_ref) - ((yb - y1) * dx_ref);
            int sign_begin = signal(res_begin);
            if (sign_begin == 0) {
                med.emplace_back((xb + x1) * 0.5f, (yb + y1) * 0.5f);
                continue;
            }

            auto xe = float(e2[end].x);
            auto ye = float(e2[end].y);
            float res_end = ((xe - x1) * dy_ref) - ((ye - y1) * dx_ref);
            int sign_end = signal(res_end);
            if (sign_end == 0) {
                med.emplace_back((xe + x1) * 0.5f, (ye + y1) * 0.5f);
                continue;
            }

            if ((sign_begin + sign_end) != 0)
                continue;

            uint j = (begin + end) >> 1;

            while (end - begin > 2) {
                auto x2 = float(e2[j].x);
                auto y2 = float(e2[j].y);
                float res = ((x2 - x1) * dy_ref) - ((y2 - y1) * dx_ref);
                int sign_res = signal(res);

                if (sign_res == 0) {
                    med.emplace_back((x2 + x1) * 0.5f, (y2 + y1) * 0.5f);
                    break;
                }

                if (sign_res + sign_begin == 0) {
                    sign_end = sign_res;
                    end = j;
                } else {
                    sign_begin = sign_res;
                    begin = j;
                }
                j = (begin + end) >> 1;
            }

            med.emplace_back((e2[j].x + x1) * 0.5f, (e2[j].y + y1) * 0.5f);
        }

        if (med.size() < 2) {
            data.isValid = false;
            return;
        }
        q4 = getMedianSlope(med, M34, data.Sb);
    }

    if (q2 == q4) {
        data.isValid = false;
        return;
    }

    float invDen = 1 / (q2 - q4);
    data.Cab.x = (M34.y - q4 * M34.x - M12.y + q2 * M12.x) * invDen;
    data.Cab.y = (q2 * M34.y - q4 * M12.y + q2 * q4 * (M12.x - M34.x)) * invDen;
    data.ta = q2;
    data.tb = q4;
    data.Ma = M12;
    data.Mb = M34;
}

Point2f
EllipseDetectorImpl::getCenterCoordinates(EllipseData &data_ij, EllipseData &data_ik) {
    float xx[7];
    float yy[7];

    xx[0] = data_ij.Cab.x;
    xx[1] = data_ik.Cab.x;
    yy[0] = data_ij.Cab.y;
    yy[1] = data_ik.Cab.y;

    {
        // 1-1
        float q2 = data_ij.ta;
        float q4 = data_ik.ta;
        Point2f &M12 = data_ij.Ma;
        Point2f &M34 = data_ik.Ma;

        float invDen = 1 / (q2 - q4);
        xx[2] = (M34.y - q4 * M34.x - M12.y + q2 * M12.x) * invDen;
        yy[2] = (q2 * M34.y - q4 * M12.y + q2 * q4 * (M12.x - M34.x)) * invDen;
    }

    {
        // 1-2
        float q2 = data_ij.ta;
        float q4 = data_ik.tb;
        Point2f &M12 = data_ij.Ma;
        Point2f &M34 = data_ik.Mb;

        float invDen = 1 / (q2 - q4);
        xx[3] = (M34.y - q4 * M34.x - M12.y + q2 * M12.x) * invDen;
        yy[3] = (q2 * M34.y - q4 * M12.y + q2 * q4 * (M12.x - M34.x)) * invDen;
    }

    {
        // 2-2
        float q2 = data_ij.tb;
        float q4 = data_ik.tb;
        Point2f &M12 = data_ij.Mb;
        Point2f &M34 = data_ik.Mb;

        float invDen = 1 / (q2 - q4);
        xx[4] = (M34.y - q4 * M34.x - M12.y + q2 * M12.x) * invDen;
        yy[4] = (q2 * M34.y - q4 * M12.y + q2 * q4 * (M12.x - M34.x)) * invDen;
    }

    {
        // 2-1
        float q2 = data_ij.tb;
        float q4 = data_ik.ta;
        Point2f &M12 = data_ij.Mb;
        Point2f &M34 = data_ik.Ma;

        float invDen = 1 / (q2 - q4);
        xx[5] = (M34.y - q4 * M34.x - M12.y + q2 * M12.x) * invDen;
        yy[5] = (q2 * M34.y - q4 * M12.y + q2 * q4 * (M12.x - M34.x)) * invDen;
    }

    xx[6] = (xx[0] + xx[1]) * 0.5f;
    yy[6] = (yy[0] + yy[1]) * 0.5f;


    // median
    std::nth_element(xx, xx + 3, xx + 7);
    std::nth_element(yy, yy + 3, yy + 7);
    float xc = xx[3];
    float yc = yy[3];

    return {xc, yc};
}


void EllipseDetectorImpl::labeling(Mat1b &image, VVP &segments, int minLength) {
    const int RG_STACK_SIZE = 2048;
    int stackInt[RG_STACK_SIZE];
    Point stackPoint[RG_STACK_SIZE];

    Mat_<uchar> image_clone = image.clone();
    Size imgSize = image.size();
    int h = imgSize.height, w = imgSize.width;

    Point point;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if ((image_clone(y, x)) != 0) {
                // for each point
                int curInt = 0;
                int i = x + y * w;
                stackInt[curInt] = i;
                curInt++;

                // clear point list
                int curPoint = 0;
                while (curInt > 0) {
                    curInt--;
                    i = stackInt[curInt];
                    int x2 = i % w;
                    int y2 = i / w;

                    point.x = x2;
                    point.y = y2;

                    if ((image_clone(y2, x2))) {
                        stackPoint[curPoint] = point;
                        curPoint++;
                        image_clone(y2, x2) = 0;
                    }

                    // 4 directions
                    if (x2 > 0 && (image_clone(y2, x2 - 1) != 0)) {
                        stackInt[curInt] = i - 1;
                        curInt++;
                    }
                    if (y2 > 0 && (image_clone(y2 - 1, x2) != 0)) {
                        stackInt[curInt] = i - w;
                        curInt++;
                    }
                    if (y2 < h - 1 && (image_clone(y2 + 1, x2) != 0)) {
                        stackInt[curInt] = i + w;
                        curInt++;
                    }
                    if (x2 < w - 1 && (image_clone(y2, x2 + 1) != 0)) {
                        stackInt[curInt] = i + 1;
                        curInt++;
                    }

                    // 8 directions
                    if (x2 > 0 && y2 > 0 && (image_clone(y2 - 1, x2 - 1) != 0)) {
                        stackInt[curInt] = i - w - 1;
                        curInt++;
                    }
                    if (x2 > 0 && y2 < h - 1 && (image_clone(y2 + 1, x2 - 1) != 0)) {
                        stackInt[curInt] = i + w - 1;
                        curInt++;
                    }
                    if (x2 < w - 1 && y2 > 0 && (image_clone(y2 - 1, x2 + 1) != 0)) {
                        stackInt[curInt] = i - w + 1;
                        curInt++;
                    }
                    if (x2 < w - 1 && y2 < h - 1 && (image_clone(y2 + 1, x2 + 1) != 0)) {
                        stackInt[curInt] = i + w + 1;
                        curInt++;
                    }
                }

                if (curPoint >= minLength) {
                    std::vector<Point> component;
                    component.reserve(curPoint);
                    for (int j = 0; j < curPoint; j++)
                        component.push_back(stackPoint[j]);
                    segments.push_back(component);
                }
            }
        }
    }
}

void EllipseDetectorImpl::detectEdges13(Mat1b &DP, VVP &points_1, VVP &points_3) {
    // vector of connected edge points
    VVP contours;
    int countEdges = 0;
    // labeling 8-connected edge points, discarding edge too small
    labeling(DP, contours, _minEdgeLength); // label point on the same arc
    int contourSize = int(contours.size());

    // for each edge
    for (int i = 0; i < contourSize; i++) {
        VP &edgeSegment = contours[i];

        // selection strategy - constraint on axes aspect ratio
        RotatedRect oriented = minAreaRect(edgeSegment);
        float orMin = min(oriented.size.width, oriented.size.height);

        if (orMin < _minOrientedRectSide) {
            countEdges++;
            continue;
        }

        // order edge points of the same arc
        sort(edgeSegment.begin(), edgeSegment.end(), sortPoint);
        int edgeSegmentSize = unsigned(edgeSegment.size());

        // get extrema of the arc
        Point &left = edgeSegment[0];
        Point &right = edgeSegment[edgeSegmentSize - 1];

        // find convexity
        int countTop = 0;
        int lx = left.x;
        for (int k = 1; k < edgeSegmentSize; ++k) {
            if (edgeSegment[k].x == lx)
                continue;
            countTop += (edgeSegment[k].y - left.y);
            lx = edgeSegment[k].x;
        }

        int width = abs(right.x - left.x) + 1;
        int height = abs(right.y - left.y) + 1;
        int countBottom = (width * height) - edgeSegmentSize - countTop;

        if (countBottom > countTop)
            points_1.push_back(edgeSegment);
        else if (countBottom < countTop)
            points_3.push_back(edgeSegment);
    }
}

void EllipseDetectorImpl::detectEdges24(Mat1b &DN, VVP &points_2, VVP &points_4) {
    // vector of connected edge points
    VVP contours;
    int countEdges = 0;
    // labeling 8-connected edge points, discarding edge too small
    labeling(DN, contours, _minEdgeLength); // label point on the same arc
    int contourSize = int(contours.size());

    // for each edge
    for (int i = 0; i < contourSize; i++) {
        VP &edgeSegment = contours[i];

        // selection strategy - constraint on axes aspect ratio
        RotatedRect oriented = minAreaRect(edgeSegment);
        float orMin = min(oriented.size.width, oriented.size.height);

        if (orMin < _minOrientedRectSide) {
            countEdges++;
            continue;
        }

        // order edge points of the same arc
        sort(edgeSegment.begin(), edgeSegment.end(), sortPoint);
        int edgeSegmentSize = unsigned(edgeSegment.size());

        // get extrema of the arc
        Point &left = edgeSegment[0];
        Point &right = edgeSegment[edgeSegmentSize - 1];

        // find convexity
        int countBottom = 0;
        int lx = left.x;
        for (int k = 0; k < edgeSegmentSize; ++k) {
            if (edgeSegment[k].x == lx)
                continue;
            countBottom += (left.y - edgeSegment[k].y);
            lx = edgeSegment[k].x;
        }

        int width = abs(right.x - left.x) + 1;
        int height = abs(right.y - left.y) + 1;
        int countTop = (width * height) - edgeSegmentSize - countBottom;

        if (countBottom > countTop)
            points_2.push_back(edgeSegment);
        else if (countBottom < countTop)
            points_4.push_back(edgeSegment);
    }
}

#define T124 pjf,pjm,pjl,pif,pim,pil
#define T231 pil,pim,pif,pjf,pjm,pjl
#define T342 pif,pim,pil,pjf,pjm,pjl
#define T413 pif,pim,pil,pjl,pjm,pjf

void EllipseDetectorImpl::getTriplets124(VVP &pi, VVP &pj, VVP &pk,
                                         std::unordered_map<uint, EllipseData> &data,
                                         std::vector<Ellipse> &ellipses) {
    // get arcs length
    auto sz_i = ushort(pi.size());
    auto sz_j = ushort(pj.size());
    auto sz_k = ushort(pk.size());

    // for each edge i
    for (ushort i = 0; i < sz_i; i++) {
        VP &edge_i = pi[i];
        auto sz_ei = ushort(edge_i.size());

        Point &pif = edge_i[0];
        Point &pim = edge_i[sz_ei / 2];
        Point &pil = edge_i[sz_ei - 1];

        // 1 -> reverse 1
        VP rev_i(edge_i.size());
        reverse_copy(edge_i.begin(), edge_i.end(), rev_i.begin());

        // for each edge j
        for (ushort j = 0; j < sz_j; ++j) {
            VP &edge_j = pj[j];
            auto sz_ej = ushort(edge_j.size());

            Point &pjf = edge_j[0];
            Point &pjm = edge_j[sz_ej / 2];
            Point &pjl = edge_j[sz_ej - 1];

            // constraints on position
            if (pjl.x > pif.x + _positionThreshold)
                continue;

            // constraints on CNC
            const float CNC_THRESHOLD = 0.3f;
            if (fabs(valueOfPoints(T124) - 1) > CNC_THRESHOLD)
                continue;

            uint key_ij = generateKey(PAIR_12, i, j);

            // for each edge k
            for (ushort k = 0; k < sz_k; ++k) {
                VP &edge_k = pk[k];
                auto sz_ek = ushort(edge_k.size());

                Point &pkl = edge_k[sz_ek - 1];

                // constraints on position
                if (pkl.y < pil.y - _positionThreshold)
                    continue;

                uint key_ik = generateKey(PAIR_14, i, k);

                // find centers
                EllipseData data_ij, data_ik;

                // if the data for the pair i-j have not been computed yet
                if (data.count(key_ij) == 0) {
                    getFastCenter(edge_j, rev_i, data_ij);
                    // insert computed data in the hash table
                    data.insert(std::pair<uint, EllipseData>(key_ij, data_ij));
                } else {
                    // otherwise, just lookup the data in the hash table
                    data_ij = data.at(key_ij);
                }

                // if the data for the pair i-k have not been computed yet
                if (data.count(key_ik) == 0) {
                    getFastCenter(edge_i, edge_k, data_ik);
                    // insert computed data in the hash table
                    data.insert(std::pair<uint, EllipseData>(key_ik, data_ik));
                } else {
                    // otherwise, just lookup the data in the hash table
                    data_ik = data.at(key_ik);
                }

                // invalid centers
                if (!data_ij.isValid || !data_ik.isValid)
                    continue;

                // selection strategy - Step 3.
                // the computed centers are not close enough
                if (pointDistance2(data_ij.Cab, data_ik.Cab) > _maxCenterDistance2)
                    continue;

                // find ellipse parameters
                // get the coordinates of the center (xc, yc)
                Point2f center = getCenterCoordinates(data_ij, data_ik);
                // find remaining parameters (A, B, rho)
                findEllipses(center, edge_i, edge_j, edge_k, data_ij, data_ik, ellipses);
            }
        }
    }
}

void EllipseDetectorImpl::getTriplets231(VVP &pi, VVP &pj, VVP &pk,
                                         std::unordered_map<uint, EllipseData> &data,
                                         std::vector<Ellipse> &ellipses) {
    // get arc length
    auto sz_i = ushort(pi.size());
    auto sz_j = ushort(pj.size());
    auto sz_k = ushort(pk.size());

    // for each edge i
    for (ushort i = 0; i < sz_i; i++) {
        VP &edge_i = pi[i];
        auto sz_ei = ushort(edge_i.size());

        Point &pif = edge_i[0];
        Point &pim = edge_i[sz_ei / 2];
        Point &pil = edge_i[sz_ei - 1];

        // 2 -> reverse 2
        VP rev_i(edge_i.size());
        reverse_copy(edge_i.begin(), edge_i.end(), rev_i.begin());

        // for each edge j
        for (ushort j = 0; j < sz_j; ++j) {
            VP &edge_j = pj[j];
            auto sz_ej = ushort(edge_j.size());

            Point &pjf = edge_j[0];
            Point &pjm = edge_j[sz_ej / 2];
            Point &pjl = edge_j[sz_ej - 1];

            // constraints on position
            if (pjf.y < pif.y - _positionThreshold)
                continue;

            // constraints on CNC
            const float CNC_THRESHOLD = 0.3f;
            if (fabs(valueOfPoints(T231) - 1) > CNC_THRESHOLD)
                continue;

            // 3 -> reverse 3
            VP rev_j(edge_j.size());
            reverse_copy(edge_j.begin(), edge_j.end(), rev_j.begin());

            uint key_ij = generateKey(PAIR_23, i, j);

            // for each edge k
            for (ushort k = 0; k < sz_k; ++k) {
                VP &edge_k = pk[k];

                Point &pkf = edge_k[0];

                // constraints on position
                if (pkf.x < pil.x - _positionThreshold)
                    continue;

                uint key_ik = generateKey(PAIR_12, k, i);

                // find centers
                EllipseData data_ij, data_ik;

                // if the data for the pair i-j have not been computed yet
                if (data.count(key_ij) == 0) {
                    getFastCenter(rev_i, rev_j, data_ij);
                    // insert computed date in the hash table
                    data.insert(std::pair<uint, EllipseData>(key_ij, data_ij));
                } else {
                    // otherwise, just lookup the data in the hash table
                    data_ij = data.at(key_ij);
                }

                // if the data for the pair i-k have not been computed yet
                if (data.count(key_ik) == 0) {
                    // 1 -> reverse 1
                    VP rev_k(edge_k.size());
                    reverse_copy(edge_k.begin(), edge_k.end(), rev_k.begin());

                    getFastCenter(edge_i, rev_k, data_ik);
                    data.insert(std::pair<uint, EllipseData>(key_ik, data_ik));
                } else {
                    // otherwise, just lookup the data in the hash table
                    data_ik = data.at(key_ik);
                }

                // invalid centers
                if (!data_ij.isValid || !data_ik.isValid)
                    continue;

                // selection strategy - Step 3.
                // the computed centers are not close enough
                if (pointDistance2(data_ij.Cab, data_ik.Cab) > _maxCenterDistance2)
                    continue;

                // find ellipse parameters
                // get the coordinates of the center (xc, yc)
                Point2f center = getCenterCoordinates(data_ij, data_ik);
                // find remaining parameters (A, B, rho)
                findEllipses(center, edge_i, edge_j, edge_k, data_ij, data_ik, ellipses);
            }
        }
    }
}

void EllipseDetectorImpl::getTriplets342(VVP &pi, VVP &pj, VVP &pk,
                                         std::unordered_map<uint, EllipseData> &data,
                                         std::vector<Ellipse> &ellipses) {
    // get arcs length
    auto sz_i = ushort(pi.size());
    auto sz_j = ushort(pj.size());
    auto sz_k = ushort(pk.size());

    // for each edge i
    for (ushort i = 0; i < sz_i; i++) {
        VP &edge_i = pi[i];
        auto sz_ei = ushort(edge_i.size());

        Point &pif = edge_i[0];
        Point &pim = edge_i[sz_ei / 2];
        Point &pil = edge_i[sz_ei - 1];

        // 3 -> reverse 3
        VP rev_i(edge_i.size());
        reverse_copy(edge_i.begin(), edge_i.end(), rev_i.begin());

        // for each edge j
        for (ushort j = 0; j < sz_j; ++j) {
            VP &edge_j = pj[j];
            auto sz_ej = ushort(edge_j.size());

            Point &pjf = edge_j[0];
            Point &pjm = edge_j[sz_ej / 2];
            Point &pjl = edge_j[sz_ej - 1];

            // constraints on position
            if (pjf.x < pil.x - _positionThreshold)
                continue;

            // constraints on CNC
            const float CNC_THRESHOLD = 0.3f;
            if (fabs(valueOfPoints(T342) - 1) > CNC_THRESHOLD)
                continue;

            // 4 -> reverse 4
            VP rev_j(edge_j.size());
            reverse_copy(edge_j.begin(), edge_j.end(), rev_j.begin());

            uint key_ij = generateKey(PAIR_34, i, j);

            // for each edge k
            for (ushort k = 0; k < sz_k; ++k) {
                VP &edge_k = pk[k];

                Point &pkf = edge_k[0];

                // constraints on position
                if (pkf.y > pif.y + _positionThreshold)
                    continue;

                uint key_ik = generateKey(PAIR_23, k, i);

                // find centers
                EllipseData data_ij, data_ik;

                // if the data for the pair i-j have not been computed yet
                if (data.count(key_ij) == 0) {
                    getFastCenter(edge_i, rev_j, data_ij);
                    // insert computed data in the hash table
                    data.insert(std::pair<uint, EllipseData>(key_ij, data_ij));
                } else {
                    // otherwise, just lookup the data in the hash table
                    data_ij = data.at(key_ij);
                }

                // if the data for the pair i-k have not been computed yet
                if (data.count(key_ik) == 0) {
                    // 2 -> reverse 2
                    VP rev_k(edge_k.size());
                    reverse_copy(edge_k.begin(), edge_k.end(), rev_k.begin());

                    getFastCenter(rev_i, rev_k, data_ik);
                    data.insert(std::pair<uint, EllipseData>(key_ik, data_ik));
                } else {
                    // otherwise, just lookup the data in the hash table
                    data_ik = data.at(key_ik);
                }

                // invalid centers
                if (!data_ij.isValid || !data_ik.isValid)
                    continue;

                // selection strategy - Step 3.
                // the computed centers are not close enough
                if (pointDistance2(data_ij.Cab, data_ik.Cab) > _maxCenterDistance2)
                    continue;

                // find ellipse parameters
                // get the coordinates of the center (xc, yc)
                Point2f center = getCenterCoordinates(data_ij, data_ik);
                // find remaining parameters (A, B, rho)
                findEllipses(center, edge_i, edge_j, edge_k, data_ij, data_ik, ellipses);
            }
        }

    }
}

void EllipseDetectorImpl::getTriplets413(VVP &pi, VVP &pj, VVP &pk,
                                         std::unordered_map<uint, EllipseData> &data,
                                         std::vector<Ellipse> &ellipses) {
    // get arch length
    auto sz_i = ushort(pi.size());
    auto sz_j = ushort(pj.size());
    auto sz_k = ushort(pk.size());

    // for each edge i
    for (ushort i = 0; i < sz_i; i++) {
        VP &edge_i = pi[i];
        auto sz_ei = ushort(edge_i.size());

        Point &pif = edge_i[0];
        Point &pim = edge_i[sz_ei / 2];
        Point &pil = edge_i[sz_ei - 1];

        // 4 -> reverse 4
        VP rev_i(edge_i.size());
        reverse_copy(edge_i.begin(), edge_i.end(), rev_i.begin());

        // for each edge j
        for (ushort j = 0; j < sz_j; ++j) {
            VP &edge_j = pj[j];
            auto sz_ej = ushort(edge_j.size());

            Point &pjf = edge_j[0];
            Point &pjm = edge_j[sz_ej / 2];
            Point &pjl = edge_j[sz_ej - 1];

            // constraints on position
            if (pjl.y > pil.y + _positionThreshold)
                continue;

            // constraints on CNC
            const float CNC_THRESHOLD = 0.3f;
            if (fabs(valueOfPoints(T413) - 1) > CNC_THRESHOLD)
                continue;

            uint key_ij = generateKey(PAIR_14, j, i);

            // for each edge k
            for (ushort k = 0; k < sz_k; ++k) {
                VP &edge_k = pk[k];
                auto sz_ek = ushort(edge_k.size());

                Point &pkl = edge_k[sz_ek - 1];

                // constraints on position
                if (pkl.x > pif.x + _positionThreshold)
                    continue;

                uint key_ik = generateKey(PAIR_34, k, i);

                // find centers
                EllipseData data_ij, data_ik;

                // if the data for the pair i-j have not been computed yet
                if (data.count(key_ij) == 0) {
                    getFastCenter(edge_i, edge_j, data_ij);
                    // insert computed date in the hash table
                    data.insert(std::pair<uint, EllipseData>(key_ij, data_ij));
                } else {
                    // otherwise, just lookup the data in the hash table
                    data_ij = data.at(key_ij);
                }

                // if the data for the pair i-k have not been computed yet
                if (data.count(key_ik) == 0) {
                    getFastCenter(rev_i, edge_k, data_ik);
                    data.insert(std::pair<uint, EllipseData>(key_ik, data_ik));
                } else {
                    // otherwise, just lookup the data in the hash table
                    data_ik = data.at(key_ik);
                }

                // invalid centers
                if (!data_ij.isValid || !data_ik.isValid)
                    continue;

                // selection strategy - Step 3.
                // the computed centers are not close enough
                if (pointDistance2(data_ij.Cab, data_ik.Cab) > _maxCenterDistance2)
                    continue;

                // find ellipse parameters
                // get the coordinates of the center (xc, yc)
                Point2f center = getCenterCoordinates(data_ij, data_ik);
                // find remaining parameters (A, B, rho)
                findEllipses(center, edge_i, edge_j, edge_k, data_ij, data_ik, ellipses);

            }
        }
    }
}

void EllipseDetectorImpl::preProcessing(Mat1b &image, Mat1b &dp, Mat1b &dn) {
    // smooth image
    GaussianBlur(image, image, _kernelSize, _sigma);

    // temp variables
    Mat1b edges;// edge mask
    Mat1s dx, dy; // sobel derivatives

    // detect edges
    Sobel(image, dx, CV_16S, 1, 0, 3, 1, 0, BORDER_REPLICATE);
    Sobel(image, dy, CV_16S, 0, 1, 3, 1, 0, BORDER_REPLICATE);

    // calculate magnitude of gradient
    Size imgSize = image.size();
    edges.create(imgSize);
    Mat1f magGrad(imgSize.height, imgSize.width, 0.f);
    float maxGrad(0);
    for (int i = 0; i < imgSize.height; i++) {
        auto *tmpMag = magGrad.ptr<float>(i);
        for (int j = 0; j < imgSize.width; j++) {
            auto val = float(abs(dx[i][j]) + abs(dy[i][j]));
            tmpMag[j] = val;
            maxGrad = (val > maxGrad) ? val : maxGrad;
        }
    }

    // set magic numbers
    const int NUM_BINS = 64;
    const double PERCENT_OF_PIXEL_NOT_EDGES = 0.9f, RATIO_THRESHOLD = 0.3f;

    // compute histogram
    int binSize = cvFloor(maxGrad / float(NUM_BINS) + 0.5f) + 1;
    binSize = max(1, binSize);
    int bins[NUM_BINS] = {0};
    for (int i = 0; i < imgSize.height; i++) {
        auto *tmpMag = magGrad.ptr<float>(i);
        for (int j = 0; j < imgSize.width; j++)
            bins[int(tmpMag[j]) / binSize]++;
    }

    // select the thresholds
    float total = 0.f;
    auto target = float(imgSize.height * imgSize.width * PERCENT_OF_PIXEL_NOT_EDGES);
    int lowTh, highTh(0);

    while (total < target) {
        total += bins[highTh];
        highTh++;
    }
    highTh = cvFloor(highTh * binSize);
    lowTh = cvFloor(RATIO_THRESHOLD * float(highTh));

    // buffer
    int *magBuffer[3];
    void *buffer = malloc((imgSize.width + 2) * (imgSize.height + 2) +
                          (imgSize.width + 2) * 3 * sizeof(int));
    magBuffer[0] = (int *) buffer;
    magBuffer[1] = magBuffer[0] + imgSize.width + 2;
    magBuffer[2] = magBuffer[1] + imgSize.width + 2;
    uchar *map = (uchar *) (magBuffer[2] + imgSize.width + 2);
    ptrdiff_t mapStep = imgSize.width + 2;

    int maxSize = MAX(1 << 10, imgSize.width * imgSize.height / 10);
    std::vector<uchar *> stack;
    stack.resize(maxSize);
    uchar **stackTop = 0, **stackBottom = 0;
    stackTop = stackBottom = &stack[0];

    memset(magBuffer[0], 0, (imgSize.width + 2) * sizeof(int));
    memset(map, 1, mapStep);
    memset(map + mapStep * (imgSize.height + 1), 1, mapStep);

    Mat magRow;
    magRow.create(1, imgSize.width, CV_32F);

    // calculate magnitude and angle of gradient, perform non-maxima suppression.
    // fill the map with one of the following values:
    // 0 - the pixel might belong to an edge
    // 1 - the pixel can not belong to an edge
    // 2 - the pixel does belong to an edge
    for (int i = 0; i <= imgSize.height; i++) {
        int *tmpMag = magBuffer[(i > 0) + 1] + 1;
        const short *tmpDx = (short *) (dx[i]);
        const short *tmpDy = (short *) (dy[i]);
        uchar *tmpMap;
        int prevFlag = 0;

        if (i < imgSize.height) {
            tmpMag[-1] = tmpMag[imgSize.width] = 0;
            for (int j = 0; j < imgSize.width; j++)
                tmpMag[j] = abs(tmpDx[j]) + abs(tmpDy[j]);
        } else
            memset(tmpMag - 1, 0, (imgSize.width + 2) * sizeof(int));

        // at the very beginning we do not have a complete ring
        // buffer of 3 magnitude rows for non-maxima suppression
        if (i == 0)
            continue;

        tmpMap = map + mapStep * i + 1;
        tmpMap[-1] = tmpMap[imgSize.width] = 1;

        tmpMag = magBuffer[1] + 1; // take the central row
        tmpDx = (short *) (dx[i - 1]);
        tmpDy = (short *) (dy[i - 1]);

        ptrdiff_t magStep1, magStep2;
        magStep1 = magBuffer[2] - magBuffer[1];
        magStep2 = magBuffer[0] - magBuffer[1];

        if ((stackTop - stackBottom) + imgSize.width > maxSize) {
            int stackSize = (int) (stackTop - stackBottom);
            maxSize = MAX(maxSize * 3 / 2, maxSize + 8);
            stack.resize(maxSize);
            stackBottom = &stack[0];
            stackTop = stackBottom + stackSize;
        }

        const int CANNY_SHIFT = 15;
        const float TAN22_5 = 0.4142135623730950488016887242097f; // tan(22.5) = sqrt(2) - 1
        const int TG22 = (int) (TAN22_5 * (1 << CANNY_SHIFT) + 0.5);

        // #define CANNY_PUSH(d)    *(d) = (uchar)2, *stack_top++ = (d)
        // #define CANNY_POP(d)     ((d) = *--stack_top)

        for (int j = 0; j < imgSize.width; j++) {
            int x = tmpDx[j], y = tmpDy[j];
            int s = x ^ y;
            int m = tmpMag[j];
            x = abs(x), y = abs(y);
            if (m > lowTh) {
                int tg22x = x * TG22;
                int tg67x = tg22x + ((x + x) << CANNY_SHIFT);

                y <<= CANNY_SHIFT;
                if (y < tg22x) {
                    if (m > tmpMag[j - 1] && m >= tmpMag[j + 1]) {
                        if (m > highTh && !prevFlag && tmpMap[j - mapStep] != 2) {
                            tmpMap[j] = (uchar) 2;
                            *stackTop++ = (tmpMap + j);
                            prevFlag = 1;
                        } else
                            tmpMap[j] = (uchar) 0;
                        continue;
                    }
                } else if (y > tg67x) {
                    if (m > tmpMag[j + magStep2] && m >= tmpMag[j + magStep1]) {
                        if (m > highTh && !prevFlag && tmpMap[j - mapStep] != 2) {
                            tmpMap[j] = (uchar) 2;
                            *stackTop++ = (tmpMap + j);
                            prevFlag = 1;
                        } else
                            tmpMap[j] = (uchar) 0;
                        continue;
                    }
                } else {
                    s = s < 0 ? -1 : 1;
                    if (m > tmpMag[j + magStep2 - s] && m > tmpMag[j + magStep1 + s]) {
                        if (m > highTh && !prevFlag && tmpMap[j - mapStep] != 2) {
                            tmpMap[j] = (uchar) 2;
                            *stackTop++ = (tmpMap + j);
                            prevFlag = 1;
                        } else
                            tmpMap[j] = (uchar) 0;
                        continue;
                    }
                }
            }
            prevFlag = 0;
            tmpMap[j] = (uchar) 1;
        }

        // scroll the ring buffer
        tmpMag = magBuffer[0];
        magBuffer[0] = magBuffer[1];
        magBuffer[1] = magBuffer[2];
        magBuffer[2] = tmpMag;
    }

    // track the edges (hysteresis thresholding)
    while (stackTop > stackBottom) {
        uchar *m;
        if ((stackTop - stackBottom) + 8 > maxSize) {
            int stackSize = (int) (stackTop - stackBottom);
            maxSize = MAX(maxSize * 3 / 2, maxSize + 8);
            stack.resize(maxSize);
            stackBottom = &stack[0];
            stackTop = stackBottom + stackSize;
        }

        m = *--stackTop;
        if (!m[-1]) {
            *(m - 1) = (uchar) 2;
            *stackTop++ = m - 1;
        }
        if (!m[1]) {
            *(m + 1) = (uchar) 2;
            *stackTop++ = m + 1;
        }
        if (!m[-mapStep - 1]) {
            *(m - mapStep - 1) = (uchar) 2;
            *stackTop++ = m - mapStep - 1;
        }
        if (!m[-mapStep]) {
            *(m - mapStep) = (uchar) 2;
            *stackTop++ = m - mapStep;
        }
        if (!m[-mapStep + 1]) {
            *(m - mapStep + 1) = (uchar) 2;
            *stackTop++ = m - mapStep + 1;
        }
        if (!m[mapStep - 1]) {
            *(m + mapStep - 1) = (uchar) 2;
            *stackTop++ = m + mapStep - 1;
        }
        if (!m[mapStep]) {
            *(m + mapStep) = (uchar) 2;
            *stackTop++ = m + mapStep;
        }
        if (!m[mapStep + 1]) {
            *(m + mapStep + 1) = (uchar) 2;
            *stackTop++ = m + mapStep + 1;
        }
    }

    // final pass, form the final image
    for (int i = 0; i < imgSize.height; i++) {
        const uchar *tmpMap = map + mapStep * (i + 1) + 1;
        uchar *tmpDst = edges[i];
        for (int j = 0; j < imgSize.width; j++)
            tmpDst[j] = (uchar) -(tmpMap[j] >> 1);
    }

    // for each edge points, compute the edge direction
    for (int i = 0; i < imgSize.height; i++) {
        auto *tmpDx = dx.ptr<short>(i);
        auto *tmpDy = dy.ptr<short>(i);
        auto *tmpE = edges.ptr<uchar>(i);
        auto *tmpDp = dp.ptr<uchar>(i);
        auto *tmpDn = dn.ptr<uchar>(i);

        for (int j = 0; j < imgSize.width; j++) {
            if (!((tmpE[j] <= 0) || (tmpDx[j] == 0) || (tmpDy[j] == 0))) {
                // angle of the tangent
                float phi = -(float(tmpDx[j])) / float(tmpDy[j]);
                // along positive or negative diagonal
                if (phi > 0)
                    tmpDp[j] = (uchar) 255;
                else if (phi < 0)
                    tmpDn[j] = (uchar) 255;
            }
        }
    }
}

void EllipseDetectorImpl::detect(Mat1b &image, std::vector<Ellipse> &ellipses) {
    countsOfFindEllipse = 0, countsOfGetFastCenter = 0;

    // set the image size
    _imgSize = image.size();

    // initialize temporary data structures
    Mat1b dp = Mat1b::zeros(_imgSize); // arcs along positive diagonal
    Mat1b dn = Mat1b::zeros(_imgSize); // arcs along negative diagonal

    // initialize accumulator dimensions
    ACC_N_SIZE = 101, ACC_R_SIZE = 180, ACC_A_SIZE = max(_imgSize.height, _imgSize.width);

    // allocate accumulators
    accN = new int[ACC_N_SIZE], accR = new int[ACC_R_SIZE], accA = new int[ACC_A_SIZE];

    // other temporary
    VVP points_1, points_2, points_3, points_4; // vector of points, one for each convexity class
    std::unordered_map<uint, EllipseData> centers; // hash map for reusing already computed EllipseData

    // preprocessing
    // find edge point with coarse convexity along positive (dp) or negative (dn) diagonal
    preProcessing(image, dp, dn);

    // detect edge and find convexity
    detectEdges13(dp, points_1, points_3);
    detectEdges24(dn, points_2, points_4);

    // find triplets
    getTriplets124(points_1, points_2, points_4, centers, ellipses);
    getTriplets231(points_2, points_3, points_1, centers, ellipses);
    getTriplets342(points_3, points_4, points_2, centers, ellipses);
    getTriplets413(points_4, points_1, points_3, centers, ellipses);

    // sort by score
    sort(ellipses.begin(), ellipses.end());

    // free accumulator memory
    delete[]accN;
    delete[]accR;
    delete[]accA;

    // cluster detections
    clusterEllipses(ellipses);
}

void EllipseDetectorImpl::findEllipses(Point2f &center, VP &edge_i, VP &edge_j, VP &edge_k,
                                       EllipseData &data_ij, EllipseData &data_ik,
                                       std::vector<Ellipse> &ellipses) {
    countsOfFindEllipse++;
    // find ellipse parameters

    // 0-initialize accumulators
    memset(accN, 0, sizeof(int) * ACC_N_SIZE);
    memset(accR, 0, sizeof(int) * ACC_R_SIZE);
    memset(accA, 0, sizeof(int) * ACC_A_SIZE);


    // get size of the 4 vectors of slopes (2 pairs of arcs)
    int sz_ij1 = int(data_ij.Sa.size());
    int sz_ij2 = int(data_ij.Sb.size());
    int sz_ik1 = int(data_ik.Sa.size());
    int sz_ik2 = int(data_ik.Sb.size());

    // get the size of the 3 arcs
    size_t sz_ei = edge_i.size();
    size_t sz_ej = edge_j.size();
    size_t sz_ek = edge_k.size();

    // center of the estimated ellipse
    float a0 = center.x;
    float b0 = center.y;

    // estimation of remaining parameters
    // uses 4 combinations of parameters. See Table 1 and Sect [3.2.3] of the paper.
    // ij1 and ik
    {
        float q1 = data_ij.ra;
        float q3 = data_ik.ra;
        float q5 = data_ik.rb;

        for (int ij1 = 0; ij1 < sz_ij1; ij1++) {
            float q2 = data_ij.Sa[ij1];

            float q1xq2 = q1 * q2;
            // ij1 and ik1
            for (int ik1 = 0; ik1 < sz_ik1; ik1++) {
                float q4 = data_ik.Sa[ik1];

                float q3xq4 = q3 * q4;

                // see Eq. [13-18] in the paper

                float a = (q1xq2 - q3xq4); // gama
                float b = (q3xq4 + 1) * (q1 + q2) - (q1xq2 + 1) * (q3 + q4); // beta
                float Kp = (-b + sqrt(b * b + 4 * a * a)) / (2 * a); // K+
                float zplus = ((q1 - Kp) * (q2 - Kp)) / ((1 + q1 * Kp) * (1 + q2 * Kp));

                // available or not
                if (zplus >= 0.0f) continue;

                float Np = sqrt(-zplus); // N+
                float rho = atan(Kp); // rho tmp
                int rhoDeg;
                if (Np > 1.f) {
                    // inverse and convert to [0, 180)
                    Np = 1.f / Np;
                    rhoDeg = cvRound((rho * 180 / CV_PI) + 180) % 180;
                } else {
                    // convert to [0, 180)
                    rhoDeg = cvRound((rho * 180 / CV_PI) + 90) % 180;
                }

                int iNp = cvRound(Np * 100);

                if (0 <= iNp && iNp < ACC_N_SIZE && 0 <= rhoDeg && rhoDeg < ACC_R_SIZE) {
                    ++accN[iNp];    // increment N accumulator
                    ++accR[rhoDeg];    // increment R accumulator
                }
            }

            // ij1 and ik2
            for (int ik2 = 0; ik2 < sz_ik2; ik2++) {
                float q4 = data_ik.Sb[ik2];

                float q5xq4 = q5 * q4;

                // See Eq. [13-18] in the paper

                float a = (q1xq2 - q5xq4);
                float b = (q5xq4 + 1) * (q1 + q2) - (q1xq2 + 1) * (q5 + q4);
                float Kp = (-b + sqrt(b * b + 4 * a * a)) / (2 * a);
                float zplus = ((q1 - Kp) * (q2 - Kp)) / ((1 + q1 * Kp) * (1 + q2 * Kp));

                if (zplus >= 0.0f)
                    continue;

                float Np = sqrt(-zplus);
                float rho = atan(Kp);
                int rhoDeg;
                if (Np > 1.f) {
                    // inverse and convert to [0, 180)
                    Np = 1.f / Np;
                    rhoDeg = cvRound((rho * 180 / CV_PI) + 180) % 180;
                } else {
                    // convert to [0, 180)
                    rhoDeg = cvRound((rho * 180 / CV_PI) + 90) % 180;
                }

                int iNp = cvRound(Np * 100); // [0, 100]

                if (0 <= iNp && iNp < ACC_N_SIZE && 0 <= rhoDeg && rhoDeg < ACC_R_SIZE) {
                    ++accN[iNp]; // increment N accumulator
                    ++accR[rhoDeg]; // increment R accumulator
                }
            }

        }
    }

    //ij2 and ik
    {
        float q1 = data_ij.rb;
        float q3 = data_ik.rb;
        float q5 = data_ik.ra;

        for (int ij2 = 0; ij2 < sz_ij2; ij2++) {
            float q2 = data_ij.Sb[ij2];

            float q1xq2 = q1 * q2;
            //ij2 and ik2
            for (int ik2 = 0; ik2 < sz_ik2; ik2++) {
                float q4 = data_ik.Sb[ik2];

                float q3xq4 = q3 * q4;

                // See Eq. [13-18] in the paper

                float a = (q1xq2 - q3xq4);
                float b = (q3xq4 + 1) * (q1 + q2) - (q1xq2 + 1) * (q3 + q4);
                float Kp = (-b + sqrt(b * b + 4 * a * a)) / (2 * a);
                float zplus = ((q1 - Kp) * (q2 - Kp)) / ((1 + q1 * Kp) * (1 + q2 * Kp));

                if (zplus >= 0.0f)
                    continue;

                float Np = sqrt(-zplus);
                float rho = atan(Kp);
                int rhoDeg;
                if (Np > 1.f) {
                    // inverse and convert to [0, 180)
                    Np = 1.f / Np;
                    rhoDeg = cvRound((rho * 180 / CV_PI) + 180) % 180;
                } else {
                    // convert to [0, 180)
                    rhoDeg = cvRound((rho * 180 / CV_PI) + 90) % 180;
                }

                int iNp = cvRound(Np * 100); // [0, 100]

                if (0 <= iNp && iNp < ACC_N_SIZE && 0 <= rhoDeg && rhoDeg < ACC_R_SIZE) {
                    ++accN[iNp]; // increment N accumulator
                    ++accR[rhoDeg]; // increment R accumulator
                }
            }

            // ij2 and ik1
            for (int ik1 = 0; ik1 < sz_ik1; ik1++) {
                float q4 = data_ik.Sa[ik1];

                float q5xq4 = q5 * q4;

                // See Eq. [13-18] in the paper

                float a = (q1xq2 - q5xq4);
                float b = (q5xq4 + 1) * (q1 + q2) - (q1xq2 + 1) * (q5 + q4);
                float Kp = (-b + sqrt(b * b + 4 * a * a)) / (2 * a);
                float zplus = ((q1 - Kp) * (q2 - Kp)) / ((1 + q1 * Kp) * (1 + q2 * Kp));

                if (zplus >= 0.0f) {
                    continue;
                }

                float Np = sqrt(-zplus);
                float rho = atan(Kp);
                int rhoDeg;
                if (Np > 1.f) {
                    // inverse and convert to [0, 180)
                    Np = 1.f / Np;
                    rhoDeg = cvRound((rho * 180 / CV_PI) + 180) % 180;
                } else {
                    // convert to [0, 180)
                    rhoDeg = cvRound((rho * 180 / CV_PI) + 90) % 180;
                }

                int iNp = cvRound(Np * 100); // [0, 100]

                if (0 <= iNp && iNp < ACC_N_SIZE && 0 <= rhoDeg && rhoDeg < ACC_R_SIZE) {
                    ++accN[iNp]; // increment N accumulator
                    ++accR[rhoDeg]; // increment R accumulator
                }
            }
        }
    }

    // find peak in N and K accumulator
    int iN = (int)std::distance(accN, std::max_element(accN, accN + ACC_N_SIZE));
    int iK = (int)std::distance(accR, std::max_element(accR, accR + ACC_R_SIZE)) + 90;

    // recover real values
    auto fK = float(iK);
    float Np = float(iN) * 0.01f;
    float rho = fK * float(CV_PI) / 180.f; // deg 2 rad
    float Kp = tan(rho);

    // estimate A. See Eq. [19 - 22] in Sect [3.2.3] of the paper
    for (ushort l = 0; l < sz_ei; ++l) {
        Point &pp = edge_i[l];
        float sk = 1.f / sqrt(Kp * Kp + 1.f); // cos rho
        float x0 = ((pp.x - a0) * sk) + (((pp.y - b0) * Kp) * sk);
        float y0 = -(((pp.x - a0) * Kp) * sk) + ((pp.y - b0) * sk);
        float Ax = sqrt((x0 * x0 * Np * Np + y0 * y0) / ((Np * Np) * (1.f + Kp * Kp)));
        int A = cvRound(abs(Ax / cos(rho)));
        if ((0 <= A) && (A < ACC_A_SIZE))
            ++accA[A];
    }

    for (ushort l = 0; l < sz_ej; ++l) {
        Point &pp = edge_j[l];
        float sk = 1.f / sqrt(Kp * Kp + 1.f);
        float x0 = ((pp.x - a0) * sk) + (((pp.y - b0) * Kp) * sk);
        float y0 = -(((pp.x - a0) * Kp) * sk) + ((pp.y - b0) * sk);
        float Ax = sqrt((x0 * x0 * Np * Np + y0 * y0) / ((Np * Np) * (1.f + Kp * Kp)));
        int A = cvRound(abs(Ax / cos(rho)));
        if ((0 <= A) && (A < ACC_A_SIZE))
            ++accA[A];
    }

    for (ushort l = 0; l < sz_ek; ++l) {
        Point &pp = edge_k[l];
        float sk = 1.f / sqrt(Kp * Kp + 1.f);
        float x0 = ((pp.x - a0) * sk) + (((pp.y - b0) * Kp) * sk);
        float y0 = -(((pp.x - a0) * Kp) * sk) + ((pp.y - b0) * sk);
        float Ax = sqrt((x0 * x0 * Np * Np + y0 * y0) / ((Np * Np) * (1.f + Kp * Kp)));
        int A = cvRound(abs(Ax / cos(rho)));
        if ((0 <= A) && (A < ACC_A_SIZE))
            ++accA[A];
    }

    // find peak in A accumulator
    int A = (int)std::distance(accA, std::max_element(accA, accA + ACC_A_SIZE));
    auto fA = float(A);

    // find B value. See Eq [23] in the paper
    float fB = abs(fA * Np);

    // got all ellipse parameters
    Ellipse ell(a0, b0, fA, fB, fmod(rho + float(CV_PI) * 2.f, float(CV_PI)));

    // get the score. See Sect [3.3.1] in the paper

    // find the number of edge pixel lying on the ellipse
    float _cos = cos(-ell.radius);
    float _sin = sin(-ell.radius);

    float invA2 = 1.f / (ell.a * ell.a);
    float invB2 = 1.f / (ell.b * ell.b);

    float invNofPoints = 1.f / float(sz_ei + sz_ej + sz_ek);
    int counter_on_perimeter = 0;

    for (ushort l = 0; l < sz_ei; ++l) {
        float tx = float(edge_i[l].x) - ell.center.x;
        float ty = float(edge_i[l].y) - ell.center.y;
        float rx = (tx * _cos - ty * _sin);
        float ry = (tx * _sin + ty * _cos);

        float h = (rx * rx) * invA2 + (ry * ry) * invB2;
        if (abs(h - 1.f) < _maxDistanceToEllipseContour)
            ++counter_on_perimeter;
    }

    for (ushort l = 0; l < sz_ej; ++l) {
        float tx = float(edge_j[l].x) - ell.center.x;
        float ty = float(edge_j[l].y) - ell.center.y;
        float rx = (tx * _cos - ty * _sin);
        float ry = (tx * _sin + ty * _cos);

        float h = (rx * rx) * invA2 + (ry * ry) * invB2;
        if (abs(h - 1.f) < _maxDistanceToEllipseContour)
            ++counter_on_perimeter;
    }

    for (ushort l = 0; l < sz_ek; ++l) {
        float tx = float(edge_k[l].x) - ell.center.x;
        float ty = float(edge_k[l].y) - ell.center.y;
        float rx = (tx * _cos - ty * _sin);
        float ry = (tx * _sin + ty * _cos);

        float h = (rx * rx) * invA2 + (ry * ry) * invB2;
        if (abs(h - 1.f) < _maxDistanceToEllipseContour)
            ++counter_on_perimeter;
    }

    // no points found on the ellipse
    if (counter_on_perimeter <= 0)
        return;

    // compute score
    float score = float(counter_on_perimeter) * invNofPoints;
    if (score < _minScore)
        return;

    // compute reliability
    float di, dj, dk;
    {
        Point2f p1(float(edge_i[0].x), float(edge_i[0].y));
        Point2f p2(float(edge_i[sz_ei - 1].x), float(edge_i[sz_ei - 1].y));
        p1.x -= ell.center.x;
        p1.y -= ell.center.y;
        p2.x -= ell.center.x;
        p2.y -= ell.center.y;
        Point2f r1((p1.x * _cos - p1.y * _sin), (p1.x * _sin + p1.y * _cos));
        Point2f r2((p2.x * _cos - p2.y * _sin), (p2.x * _sin + p2.y * _cos));
        di = abs(r2.x - r1.x) + abs(r2.y - r1.y);
    }
    {
        Point2f p1(float(edge_j[0].x), float(edge_j[0].y));
        Point2f p2(float(edge_j[sz_ej - 1].x), float(edge_j[sz_ej - 1].y));
        p1.x -= ell.center.x;
        p1.y -= ell.center.y;
        p2.x -= ell.center.x;
        p2.y -= ell.center.y;
        Point2f r1((p1.x * _cos - p1.y * _sin), (p1.x * _sin + p1.y * _cos));
        Point2f r2((p2.x * _cos - p2.y * _sin), (p2.x * _sin + p2.y * _cos));
        dj = abs(r2.x - r1.x) + abs(r2.y - r1.y);
    }
    {
        Point2f p1(float(edge_k[0].x), float(edge_k[0].y));
        Point2f p2(float(edge_k[sz_ek - 1].x), float(edge_k[sz_ek - 1].y));
        p1.x -= ell.center.x;
        p1.y -= ell.center.y;
        p2.x -= ell.center.x;
        p2.y -= ell.center.y;
        Point2f r1((p1.x * _cos - p1.y * _sin), (p1.x * _sin + p1.y * _cos));
        Point2f r2((p2.x * _cos - p2.y * _sin), (p2.x * _sin + p2.y * _cos));
        dk = abs(r2.x - r1.x) + abs(r2.y - r1.y);
    }

    // this allows to get rid of thick edges
    float rel = min(1.f, ((di + dj + dk) / (3 * (ell.a + ell.b))));

    if (rel < _minReliability)
        return;

    // assign the new score
    ell.score = (score + rel) * 0.5f;

    // the tentative detection has been confirmed
    ellipses.emplace_back(ell);
}

void EllipseDetectorImpl::clusterEllipses(std::vector<Ellipse> &ellipses) {
    const float aDistanceThreshold = 0.1f, bDistanceThreshold = 0.1f;
    const float rDistanceThreshold = 0.1f;
    const float DcRatioThreshold = 0.1f, rCircleThreshold = 0.9f;

    int ellipseCount = int(ellipses.size());
    if (ellipseCount == 0)
        return;

    // the first ellipse is assigned to a cluster
    std::vector<Ellipse> clusters;
    clusters.emplace_back(ellipses[0]);

    bool foundCluster = false;
    for (int i = 0; i < ellipseCount; i++) {
        int clusterSize = int(clusters.size());

        Ellipse &e1 = ellipses[i];
        float ba_e1 = e1.b / e1.a;

        foundCluster = false;
        for (int j = 0; j < clusterSize; j++) {
            Ellipse &e2 = clusters[j];
            float ba_e2 = e2.b / e2.a;

            float cDistanceThreshold = min(e1.b, e2.b) * DcRatioThreshold;
            cDistanceThreshold = cDistanceThreshold * cDistanceThreshold;

            // filter centers
            float cDistance = ((e1.center.x - e2.center.x) * (e1.center.x - e2.center.x) +
                               (e1.center.y - e2.center.y) * (e1.center.y - e2.center.y));
            if (cDistance > cDistanceThreshold)
                continue;

            // filter a
            float aDistance = abs(e1.a - e2.a) / max(e1.a, e2.a);
            if (aDistance > aDistanceThreshold)
                continue;

            // filter b
            float bDistance = abs(e1.b - e2.b) / min(e1.b, e2.b);
            if (bDistance > bDistanceThreshold)
                continue;

            // filter angle
            float rDistance = getMinAnglePI(e1.radius, e2.radius) / float(CV_PI);
            if ((rDistance > rDistanceThreshold) && (ba_e1 < rCircleThreshold) &&
                (ba_e2 < rCircleThreshold))
                continue;

            // same cluster as e2
            foundCluster = true;
            break;
        }

        // create a new cluster
        if (!foundCluster)
            clusters.push_back(e1);
    }
    clusters.swap(ellipses);
}

// find ellipses in images
void findEllipses(
        InputArray image, OutputArray ellipses,
        float scoreThreshold, float reliabilityThreshold,
        float centerDistanceThreshold) {

    // check image empty and type
    CV_Assert(
            !image.empty() && (image.isMat() || image.isUMat()));

    // check ellipses type
    int type = CV_32FC(6);
    if (ellipses.fixedType()) {
        type = ellipses.type();
        CV_CheckType(type, type == CV_32FC(6), "Wrong type of output ellipses");
    }

    // set class parameters
    Size imgSize = image.size();
    float maxCenterDistance =
            sqrt(float(imgSize.width * imgSize.width + imgSize.height * imgSize.height)) *
            centerDistanceThreshold;
    EllipseDetectorImpl edi;
    edi.setParameters(maxCenterDistance, scoreThreshold, reliabilityThreshold);

    // detect - ellipse format
    std::vector<Ellipse> ellipseResults;
    Mat1b grayImage = image.getMat();
    if (image.channels() != 1)
        cvtColor(image, grayImage, COLOR_BGR2GRAY);
    edi.detect(grayImage, ellipseResults);

    // convert - ellipse format to std::vector<Vec6f>
    std::vector<Vec6f> _ellipses;
    for (size_t i = 0; i < ellipseResults.size(); i++) {
        Ellipse tmpEll = ellipseResults[i];
        Vec6f tmpVec(tmpEll.center.x, tmpEll.center.y, tmpEll.a, tmpEll.b, tmpEll.score,
                     tmpEll.radius);
        _ellipses.push_back(tmpVec);
    }
    Mat(_ellipses).copyTo(ellipses);
}
} // namespace ximgproc
} // namespace cv