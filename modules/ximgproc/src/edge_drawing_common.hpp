// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_EDGE_DRAWING_COMMON_HPP__
#define __OPENCV_EDGE_DRAWING_COMMON_HPP__

#include <opencv2/core.hpp>

#define EDGE_VERTICAL   1
#define EDGE_HORIZONTAL 2

#define ANCHOR_PIXEL  254
#define EDGE_PIXEL    255

#define LEFT  1
#define RIGHT 2
#define UP    3
#define DOWN  4

#define SOUTH_SOUTH 0
#define SOUTH_EAST 1
#define EAST_SOUTH 2
#define EAST_EAST 3

// Circular arc, circle thresholds
#define VERY_SHORT_ARC_ERROR     0.40  // Used for very short arcs (>= CANDIDATE_CIRCLE_RATIO1 && < CANDIDATE_CIRCLE_RATIO2)
#define SHORT_ARC_ERROR          1.00  // Used for short arcs (>= CANDIDATE_CIRCLE_RATIO2 && < HALF_CIRCLE_RATIO)
#define HALF_ARC_ERROR           1.25  // Used for arcs with length (>=HALF_CIRCLE_RATIO && < FULL_CIRCLE_RATIO)
#define LONG_ARC_ERROR           1.50  // Used for long arcs (>= FULL_CIRCLE_RATIO)

#define CANDIDATE_CIRCLE_RATIO1  0.25  // 25% -- If only 25% of the circle is detected, it may be a candidate for validation
#define CANDIDATE_CIRCLE_RATIO2  0.33  // 33% -- If only 33% of the circle is detected, it may be a candidate for validation
#define HALF_CIRCLE_RATIO        0.50  // 50% -- If 50% of a circle is detected at any point during joins, we immediately make it a candidate
#define FULL_CIRCLE_RATIO        0.67  // 67% -- If 67% of the circle is detected, we assume that it is fully covered

// Ellipse thresholds
#define CANDIDATE_ELLIPSE_RATIO  0.50  // 50% -- If 50% of the ellipse is detected, it may be candidate for validation
#define ELLIPSE_ERROR            1.50  // Used for ellipses. (used to be 1.65 for what reason?)
#define MAX_GRAD_VALUE 128*256

using namespace std;
using namespace cv;

class NFALUT
{
public:

    NFALUT(int size, double _prob, int _w, int _h);
    ~NFALUT();

    int* LUT; // look up table
    int LUTSize;

    double prob;
    int w, h;

    bool checkValidationByNFA(int n, int k);
    static double myAtan2(double yy, double xx);

private:
    double nfa(int n, int k);
    static double Comb(double n, double k);
};

NFALUT::NFALUT(int size, double _prob, int _w, int _h)
{
    LUTSize = size > 60 ? 60 : size;
    LUT = new int[LUTSize];
    w = _w;
    h = _h;
    prob = _prob;

    LUT[0] = 1;
    int j = 1;
    for (int i = 1; i < LUTSize; i++)
    {
        LUT[i] = LUTSize + 1;
        double ret = nfa(i, j);
        if (ret >= 1.0)
        {
            while (j < i)
            {
                j++;
                ret = nfa(i, j);
                if (ret <= 1.0)
                    break;
            }

            if (ret >= 1.0)
                continue;
        }
        LUT[i] = j;
    }
}


NFALUT::~NFALUT()
{
    delete[] LUT;
}

bool NFALUT::checkValidationByNFA(int n, int k)
{
    if (n >= LUTSize)
        return nfa(n, k) <= 1.0;
    else
        return k >= LUT[n];
}

double NFALUT::myAtan2(double yy, double xx)
{
    double angle = fastAtan2((float)yy, (float)xx);

    if (angle > 180)
        angle = angle - 180;

    return angle / 180 * CV_PI;
}

double NFALUT::Comb(double n, double k)   //fast combination computation
{
    if (k > n)
        return 0;

    double r = 1;
    for (double d = 1; d <= k; ++d)
    {
        r *= n--;
        r /= d;
    }
    return r;
}

double NFALUT::nfa(int n, int k)
{
    double sum = 0;
    double p = 0.125;
    for (int i = k; i <= n; i++)
        sum += Comb(n, i) * pow(p, i) * pow(1 - p, n - i);

    return sum * w * w * h * h;
}

struct StackNode
{
    int r, c;         // starting pixel
    int parent;       // parent chain (-1 if no parent)
    int dir;          // direction where you are supposed to go
};

// Used during Edge Linking
struct Chain
{
    int dir;          // Direction of the chain
    int len;          // # of pixels in the chain
    int parent;       // Parent of this node (-1 if no parent)
    int children[2];  // Children of this node (-1 if no children)
    Point *pixels;    // Pointer to the beginning of the pixels array
};

// light weight struct for Start & End coordinates of the line segment
struct LS
{
    Point2d start;
    Point2d end;

    LS(Point2d _start, Point2d _end)
    {
        start = _start;
        end = _end;
    }
};


struct EDLineSegment
{
    double a, b;          // y = a + bx (if invert = 0) || x = a + by (if invert = 1)
    int invert;

    double sx, sy;        // starting x & y coordinates
    double ex, ey;        // ending x & y coordinates

    int segmentNo;        // Edge segment that this line belongs to
    int firstPixelIndex;  // Index of the first pixel within the segment of pixels
    int len;              // No of pixels making up the line segment

    EDLineSegment(double _a, double _b, int _invert, double _sx, double _sy, double _ex, double _ey, int _segmentNo, int _firstPixelIndex, int _len)
    {
        a = _a;
        b = _b;
        invert = _invert;
        sx = _sx;
        sy = _sy;
        ex = _ex;
        ey = _ey;
        segmentNo = _segmentNo;
        firstPixelIndex = _firstPixelIndex;
        len = _len;
    }
};

// Circle equation: (x-xc)^2 + (y-yc)^2 = r^2
struct mCircle {
	Point2d center;
	double r;
	mCircle(Point2d _center, double _r) { center = _center; r = _r; }
};

// Ellipse equation: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
struct mEllipse {
	Point2d center;
	Size axes;
	double theta;
	mEllipse(Point2d _center, Size _axes, double _theta) { center = _center; axes = _axes; theta = _theta; }
};

//----------------------------------------------------------
// Ellipse Equation is of the form:
// Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
//
struct EllipseEquation {
	double coeff[7];  // coeff[1] = A

	EllipseEquation() {
		for (int i = 0; i<7; i++) coeff[i] = 0;
	}

	double A() { return coeff[1]; }
	double B() { return coeff[2]; }
	double C() { return coeff[3]; }
	double D() { return coeff[4]; }
	double E() { return coeff[5]; }
	double F() { return coeff[6]; }
};

// ================================ CIRCLES ================================
struct Circle {
	double xc, yc, r;        // Center (xc, yc) & radius.
	double circleFitError;   // circle fit error
	double coverRatio;       // Percentage of the circle covered by the arcs making up this circle [0-1]

	double *x, *y;           // Pointers to buffers containing the pixels making up this circle
	int noPixels;            // # of pixels making up this circle

							 // If this circle is better approximated by an ellipse, we set isEllipse to true & eq contains the ellipse's equation
	EllipseEquation eq;
	double ellipseFitError;  // ellipse fit error
	bool isEllipse;
	double majorAxisLength;     // Length of the major axis
	double minorAxisLength;     // Length of the minor axis
};

// ------------------------------------------- ARCS ----------------------------------------------------
struct MyArc {
	double xc, yc, r;           // center x, y and radius
	double circleFitError;      // Error during circle fit

	double sTheta, eTheta;      // Start & end angle in radius
	double coverRatio;          // Ratio of the pixels covered on the covering circle [0-1] (noPixels/circumference)

	int turn;                   // Turn direction: 1 or -1

	int segmentNo;              // SegmentNo where this arc belongs

	int sx, sy;                 // Start (x, y) coordinate
	int ex, ey;                 // End (x, y) coordinate of the arc

	double *x, *y;              // Pointer to buffer containing the pixels making up this arc
	int noPixels;               // # of pixels making up the arc

	bool isEllipse;             // Did we fit an ellipse to this arc?
	EllipseEquation eq;         // If an ellipse, then the ellipse's equation
	double ellipseFitError;     // Error during ellipse fit
};

// =============================== AngleSet ==================================

// add a circular arc to the list of arcs
inline double ArcLength(double sTheta, double eTheta)
{
	if (eTheta > sTheta)
        return eTheta - sTheta;
	else
        return CV_2PI - sTheta + eTheta;
}

// A fast implementation of the AngleSet class. The slow implementation is really bad. About 10 times slower than this!
struct AngleSetArc {
	double sTheta;
	double eTheta;
	int next;         // Next AngleSetArc in the linked list
};

struct AngleSet {
	AngleSetArc angles[360];
	int head;
	int next;   // Next AngleSetArc to be allocated
	double overlapAmount;   // Total overlap of the arcs in angleSet. Computed during set() function

	AngleSet() { clear(); }
	void clear() { head = -1; next = 0; overlapAmount = 0; }
	double overlapRatio() { return overlapAmount / CV_2PI; }

	void _set(double sTheta, double eTheta);
	void set(double sTheta, double eTheta);

	double _overlap(double sTheta, double eTheta);
	double overlap(double sTheta, double eTheta);

	void computeStartEndTheta(double& sTheta, double& eTheta);
	double coverRatio();
};

void AngleSet::_set(double sTheta, double eTheta)
{
    int arc = next++;

    angles[arc].sTheta = sTheta;
    angles[arc].eTheta = eTheta;
    angles[arc].next = -1;

    // Add the current arc to the linked list
    int prev = -1;
    int current = head;
    while (1)
    {
        // Empty list?
        if (head < 0)
        {
            head = arc;
            break;
        }

        // End of the list. Add to the end
        if (current < 0)
        {
            CV_Assert(prev >= 0);
            angles[prev].next = arc;
            break;
        }

        if (angles[arc].eTheta <= angles[current].sTheta)
        {
            // Add before current
            if (prev < 0)
            {
                angles[arc].next = current;
                head = arc;
            }
            else
            {
                angles[arc].next = current;
                angles[prev].next = arc;
            }
            break;
        }
        else if (angles[arc].sTheta >= angles[current].eTheta)
        {
            // continue
            prev = current;
            current = angles[current].next;

            // End of the list?
            if (current < 0)
            {
                angles[prev].next = arc;
                break;
            }
        }
        else
        {
            // overlaps with current. Join
            // First delete current from the list
            if (prev < 0)
                head = angles[head].next;
            else
                angles[prev].next = angles[current].next;

            // Update overlap amount.
            if (angles[arc].eTheta < angles[current].eTheta)
            {
                overlapAmount += angles[arc].eTheta - angles[current].sTheta;
            }
            else
            {
                overlapAmount += angles[current].eTheta - angles[arc].sTheta;
            }

            // Now join current with arc
            if (angles[current].sTheta < angles[arc].sTheta)
                angles[arc].sTheta = angles[current].sTheta;
            if (angles[current].eTheta > angles[arc].eTheta)
                angles[arc].eTheta = angles[current].eTheta;
            current = angles[current].next;
        }
    }
}

void AngleSet::set(double sTheta, double eTheta)
{
    if (eTheta > sTheta)
    {
        _set(sTheta, eTheta);
    }
    else
    {
        _set(sTheta, CV_2PI);
        _set(0, eTheta);
    }
}

double AngleSet::_overlap(double sTheta, double eTheta)
{
    double o = 0;

    int current = head;
    while (current >= 0)
    {
        if (sTheta > angles[current].eTheta)
        {
            current = angles[current].next;
            continue;
        }
        else if (eTheta < angles[current].sTheta)
            break;

        // 3 cases.
        if (sTheta < angles[current].sTheta && eTheta > angles[current].eTheta)
        {
            o += angles[current].eTheta - angles[current].sTheta;

        }
        else if (sTheta < angles[current].sTheta)
        {
            o += eTheta - angles[current].sTheta;
        }
        else
        {
            o += angles[current].eTheta - sTheta;
        }

        current = angles[current].next;
    }
    return o;
}

double AngleSet::overlap(double sTheta, double eTheta)
{
    double o;

    if (eTheta > sTheta)
    {
        o = _overlap(sTheta, eTheta);
    }
    else
    {
        o = _overlap(sTheta, CV_2PI);
        o += _overlap(0, eTheta);
    }
    return o / ArcLength(sTheta, eTheta);
}

void AngleSet::computeStartEndTheta(double& sTheta, double& eTheta)
{
    // Special case: Just one arc
    if (angles[head].next < 0)
    {
        sTheta = angles[head].sTheta;
        eTheta = angles[head].eTheta;

        return;
    }

    // OK. More than one arc. Find the biggest gap
    int current = head;
    int nextArc = angles[current].next;

    double biggestGapSTheta = angles[current].eTheta;
    double biggestGapEtheta = angles[nextArc].sTheta;
    double biggestGapLength = biggestGapEtheta - biggestGapSTheta;

    double start, end, len;
    while (1)
    {
        current = nextArc;
        nextArc = angles[nextArc].next;
        if (nextArc < 0)
            break;

        start = angles[current].eTheta;
        end = angles[nextArc].sTheta;
        len = end - start;

        if (len > biggestGapLength)
        {
            biggestGapSTheta = start;
            biggestGapEtheta = end;
            biggestGapLength = len;
        }
    }

    // Compute the gap between the last arc & the first arc
    start = angles[current].eTheta;
    end = angles[head].sTheta;
    len = CV_2PI - start + end;
    if (len > biggestGapLength)
    {
        biggestGapSTheta = start;
        biggestGapEtheta = end;
    }
    sTheta = biggestGapEtheta;
    eTheta = biggestGapSTheta;
}

double AngleSet::coverRatio()
{
    int current = head;

    double total = 0;
    while (current >= 0)
    {
        total += angles[current].eTheta - angles[current].sTheta;
        current = angles[current].next;
    }
    return total / CV_2PI;
}

struct EDArcs {
	MyArc *arcs;
	int noArcs;

public:
	EDArcs(int size = 10000) {
		arcs = new MyArc[size];
		noArcs = 0;
	}

	~EDArcs() {
		delete arcs;
	}
};

struct BufferManager {
	double *x, *y;
	int index;

	BufferManager(int maxSize) {
		x = new double[maxSize];
		y = new double[maxSize];
		index = 0;
	}

	~BufferManager() {
		delete x;
		delete y;
	}

	double *getX() { return &x[index]; }
	double *getY() { return &y[index]; }
	void move(int size) { index += size; }
};

struct Info {
	int sign;     // -1 or 1: sign of the cross product
	double angle; // angle with the next line (in radians)
	bool taken;   // Is this line taken during arc detection
};

#endif
