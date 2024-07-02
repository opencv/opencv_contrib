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

#define ED_LEFT  1
#define ED_RIGHT 2
#define ED_UP    3
#define ED_DOWN  4

#define ED_SOUTH_SOUTH 0
#define ED_SOUTH_EAST 1
#define ED_EAST_SOUTH 2
#define ED_EAST_EAST 3

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
#define ED_MAX_GRAD_VALUE 128*256

using namespace std;
using namespace cv;

#define TABSIZE 100000
#define MAX_LUT_SIZE 1024
#define RELATIVE_ERROR_FACTOR 100.0

class NFALUT
{
public:

    NFALUT(int size, double _prob, double _logNT);
    ~NFALUT();

    int* LUT; // look up table
    int LUTSize;

    double prob;
    double logNT;

    bool checkValidationByNFA(int n, int k);
    static double myAtan2(double yy, double xx);

private:
    double nfa(int n, int k);
    static double log_gamma_lanczos(double x);
    static double log_gamma_windschitl(double x);
    static double log_gamma(double x);
    static int double_equal(double a, double b);
};

NFALUT::NFALUT(int size, double _prob, double _logNT)
{
    LUTSize = size > 60 ? 60 : size;
    LUT = new int[LUTSize];
    prob = _prob;
    logNT = _logNT;

    LUT[0] = 1;
    int j = 1;
    for (int i = 1; i < LUTSize; i++)
    {
        LUT[i] = LUTSize + 1;
        double ret = nfa(i, j);
        if (ret < 0)
        {
            while (j < i)
            {
                j++;
                ret = nfa(i, j);
                if (ret >= 0)
                    break;
            }

            if (ret < 0)
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
        return nfa(n, k) >= 0.0;
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

double NFALUT::nfa(int n, int k)
{
    static double inv[TABSIZE];   /* table to keep computed inverse values */
    double tolerance = 0.1;       /* an error of 10% in the result is accepted */
    double log1term, term, bin_term, mult_term, bin_tail, err, p_term;
    int i;

    /* check parameters */
    if (n < 0 || k<0 || k>n || prob <= 0.0 || prob >= 1.0) return -1.0;

    /* trivial cases */
    if (n == 0 || k == 0) return -logNT;
    if (n == k) return -logNT - (double)n * log10(prob);

    /* probability term */
    p_term = prob / (1.0 - prob);

    /* compute the first term of the series */
    /*
    binomial_tail(n,k,p) = sum_{i=k}^n bincoef(n,i) * p^i * (1-p)^{n-i}
    where bincoef(n,i) are the binomial coefficients.
    But
    bincoef(n,k) = gamma(n+1) / ( gamma(k+1) * gamma(n-k+1) ).
    We use this to compute the first term. Actually the log of it.
    */
    log1term = log_gamma((double)n + 1.0) - log_gamma((double)k + 1.0)
        - log_gamma((double)(n - k) + 1.0)
        + (double)k * log(prob) + (double)(n - k) * log(1.0 - prob);
    term = exp(log1term);

    /* in some cases no more computations are needed */
    if (double_equal(term, 0.0)) {              /* the first term is almost zero */
        if ((double)k > (double)n * prob)     /* at begin or end of the tail?  */
            return -log1term / M_LN10 - logNT;  /* end: use just the first term  */
        else
            return -logNT;                      /* begin: the tail is roughly 1  */
    }

      /* compute more terms if needed */
    bin_tail = term;
    for (i = k + 1; i <= n; i++) {
        /*
        As
        term_i = bincoef(n,i) * p^i * (1-p)^(n-i)
        and
        bincoef(n,i)/bincoef(n,i-1) = n-1+1 / i,
        then,
        term_i / term_i-1 = (n-i+1)/i * p/(1-p)
        and
        term_i = term_i-1 * (n-i+1)/i * p/(1-p).
        1/i is stored in a table as they are computed,
        because divisions are expensive.
        p/(1-p) is computed only once and stored in 'p_term'.
        */
        bin_term = (double)(n - i + 1) * (i < TABSIZE ?
            (inv[i] != 0.0 ? inv[i] : (inv[i] = 1.0 / (double)i)) :
            1.0 / (double)i);

        mult_term = bin_term * p_term;
        term *= mult_term;
        bin_tail += term;

        if (bin_term < 1.0) {
            /* When bin_term<1 then mult_term_j<mult_term_i for j>i.
            Then, the error on the binomial tail when truncated at
            the i term can be bounded by a geometric series of form
            term_i * sum mult_term_i^j.                            */
            err = term * ((1.0 - pow(mult_term, (double)(n - i + 1))) /
                (1.0 - mult_term) - 1.0);

            /* One wants an error at most of tolerance*final_result, or:
            tolerance * abs(-log10(bin_tail)-logNT).
            Now, the error that can be accepted on bin_tail is
            given by tolerance*final_result divided by the derivative
            of -log10(x) when x=bin_tail. that is:
            tolerance * abs(-log10(bin_tail)-logNT) / (1/bin_tail)
            Finally, we truncate the tail if the error is less than:
            tolerance * abs(-log10(bin_tail)-logNT) * bin_tail        */
            if (err < tolerance * fabs(-log10(bin_tail) - logNT) * bin_tail) break;
        }
    }

    return -log10(bin_tail) - logNT;
}

double NFALUT::log_gamma_lanczos(double x)
{
    static double q[7] = { 75122.6331530, 80916.6278952, 36308.2951477,
        8687.24529705, 1168.92649479, 83.8676043424,
        2.50662827511 };
    double a = (x + 0.5) * log(x + 5.5) - (x + 5.5);
    double b = 0.0;
    int n;

    for (n = 0; n < 7; n++)
    {
        a -= log(x + (double)n);
        b += q[n] * pow(x, (double)n);
    }
    return a + log(b);
}

double NFALUT::log_gamma_windschitl(double x)
{
    return 0.918938533204673 + (x - 0.5) * log(x) - x
        + 0.5 * x * log(x * sinh(1 / x) + 1 / (810.0 * pow(x, 6.0)));
}

double NFALUT::log_gamma(double x)
{
    return x > 15 ? log_gamma_windschitl(x) : log_gamma_lanczos(x);
}

int NFALUT::double_equal(double a, double b)
{
    double abs_diff, aa, bb, abs_max;

    /* trivial case */
    if (a == b) return true;

    abs_diff = fabs(a - b);
    aa = fabs(a);
    bb = fabs(b);
    abs_max = aa > bb ? aa : bb;

    /* DBL_MIN is the smallest normalized number, thus, the smallest
    number whose relative error is bounded by DBL_EPSILON. For
    smaller numbers, the same quantization steps as for DBL_MIN
    are used. Then, for smaller numbers, a meaningful "relative"
    error should be computed by dividing the difference by DBL_MIN. */
    if (abs_max < DBL_MIN) abs_max = DBL_MIN;

    /* equal if relative error <= factor x eps */
    return (abs_diff / abs_max) <= (RELATIVE_ERROR_FACTOR * DBL_EPSILON);
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
    static constexpr int COEFF_SIZE = 7;
    double coeff[COEFF_SIZE];  // coeff[1] = A

    // Constructor using an initialization list
    EllipseEquation() : coeff{} {}

    // Accessor functions marked as const
    double A() const { return coeff[1]; }
    double B() const { return coeff[2]; }
    double C() const { return coeff[3]; }
    double D() const { return coeff[4]; }
    double E() const { return coeff[5]; }
    double F() const { return coeff[6]; }
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
		delete[] arcs;
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
		delete[] x;
		delete[] y;
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
