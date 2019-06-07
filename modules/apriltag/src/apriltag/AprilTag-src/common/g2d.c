/* Copyright (C) 2013-2016, The Regents of The University of Michigan.
All rights reserved.
This software was developed in the APRIL Robotics Lab under the
direction of Edwin Olson, ebolson@umich.edu. This software may be
available under alternative licensing terms; contact the address above.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the Regents of The University of Michigan.
*/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "g2d.h"
#include "common/math_util.h"

#ifdef _WIN32
static inline long int random(void)
{
        return rand();
}
#endif

double g2d_distance(const double a[2], const double b[2])
{
    return sqrtf(sq(a[0]-b[0]) + sq(a[1]-b[1]));
}

zarray_t *g2d_polygon_create_empty()
{
    return zarray_create(sizeof(double[2]));
}

void g2d_polygon_add(zarray_t *poly, double v[2])
{
    zarray_add(poly, v);
}

zarray_t *g2d_polygon_create_data(double v[][2], int sz)
{
    zarray_t *points = g2d_polygon_create_empty();

    for (int i = 0; i < sz; i++)
        g2d_polygon_add(points, v[i]);

    return points;
}

zarray_t *g2d_polygon_create_zeros(int sz)
{
    zarray_t *points = zarray_create(sizeof(double[2]));

    double z[2] = { 0, 0 };

    for (int i = 0; i < sz; i++)
        zarray_add(points, z);

    return points;
}

void g2d_polygon_make_ccw(zarray_t *poly)
{
    // Step one: we want the points in counter-clockwise order.
    // If the points are in clockwise order, we'll reverse them.
    double total_theta = 0;
    double last_theta = 0;

    // Count the angle accumulated going around the polygon. If
    // the sum is +2pi, it's CCW. Otherwise, we'll get -2pi.
    int sz = zarray_size(poly);

    for (int i = 0; i <= sz; i++) {
        double p0[2], p1[2];
        zarray_get(poly, i % sz, &p0);
        zarray_get(poly, (i+1) % sz, &p1);

        double this_theta = atan2(p1[1]-p0[1], p1[0]-p0[0]);

        if (i > 0) {
            double dtheta = mod2pi(this_theta-last_theta);
            total_theta += dtheta;
        }

        last_theta = this_theta;
    }

    int ccw = (total_theta > 0);

    // reverse order if necessary.
    if (!ccw) {
        for (int i = 0; i < sz / 2; i++) {
            double a[2], b[2];

            zarray_get(poly, i, a);
            zarray_get(poly, sz-1-i, b);
            zarray_set(poly, i, b, NULL);
            zarray_set(poly, sz-1-i, a, NULL);
        }
    }
}

int g2d_polygon_contains_point_ref(const zarray_t *poly, double q[2])
{
    // use winding. If the point is inside the polygon, we'll wrap
    // around it (accumulating 6.28 radians). If we're outside the
    // polygon, we'll accumulate zero.
    int psz = zarray_size(poly);

    double acc_theta = 0;

    double last_theta;

    for (int i = 0; i <= psz; i++) {
        double p[2];

        zarray_get(poly, i % psz, &p);

        double this_theta = atan2(q[1]-p[1], q[0]-p[0]);

        if (i != 0)
            acc_theta += mod2pi(this_theta - last_theta);

        last_theta = this_theta;
    }

    return acc_theta > M_PI;
}

/*
// sort by x coordinate, ascending
static int g2d_convex_hull_sort(const void *_a, const void *_b)
{
    double *a = (double*) _a;
    double *b = (double*) _b;

    if (a[0] < b[0])
        return -1;
    if (a[0] == b[0])
        return 0;
    return 1;
}
*/

/*
zarray_t *g2d_convex_hull2(const zarray_t *points)
{
    zarray_t *hull = zarray_copy(points);

    zarray_sort(hull, g2d_convex_hull_sort);

    int hsz = zarray_size(hull);
    int hout = 0;

    for (int hin = 1; hin < hsz; hin++) {
        double *p;
        zarray_get_volatile(hull, i, &p);

        // Everything to the right of hin is already convex. We now
        // add one point, p, which begins "connected" by two
        // (coincident) edges from the last right-most point to p.
        double *last;
        zarray_get_volatile(hull, hout, &last);

        // We now remove points from the convex hull by moving
    }

    return hull;
}
*/

// creates and returns a zarray(double[2]). The resulting polygon is
// CCW and implicitly closed. Unnecessary colinear points are omitted.
zarray_t *g2d_convex_hull(const zarray_t *points)
{
    zarray_t *hull = zarray_create(sizeof(double[2]));

    // gift-wrap algorithm.

    // step 1: find left most point.
    int insz = zarray_size(points);

    // must have at least 2 points. (XXX need 3?)
    assert(insz >= 2);

    double *pleft = NULL;
    for (int i = 0; i < insz; i++) {
        double *p;
        zarray_get_volatile(points, i, &p);

        if (pleft == NULL || p[0] < pleft[0])
            pleft = p;
    }

    // cannot be NULL since there must be at least one point.
    assert(pleft != NULL);

    zarray_add(hull, pleft);

    // step 2. gift wrap. Keep searching for points that make the
    // smallest-angle left-hand turn. This implementation is carefully
    // written to use only addition/subtraction/multiply. No division
    // or sqrts. This guarantees exact results for integer-coordinate
    // polygons (no rounding/precision problems).
    double *p = pleft;

    while (1) {
        assert(p != NULL);

        double *q = NULL;
        double n0 = 0, n1 = 0; // the normal to the line (p, q) (not
                       // necessarily unit length).

        // Search for the point q for which the line (p,q) is most "to
        // the right of" the other points. (i.e., every time we find a
        // point that is to the right of our current line, we change
        // lines.)
        for (int i = 0; i < insz; i++) {
            double *thisq;
            zarray_get_volatile(points, i, &thisq);

            if (thisq == p)
                continue;

            // the first time we find another point, we initialize our
            // value of q, forming the line (p,q)
            if (q == NULL) {
                q = thisq;
                n0 = q[1] - p[1];
                n1 = -q[0] + p[0];
            } else {
                // we already have a line (p,q). is point thisq RIGHT OF line (p, q)?
                double e0 = thisq[0] - p[0], e1 = thisq[1] - p[1];
                double dot = e0*n0 + e1*n1;

                if (dot > 0) {
                    // it is. change our line.
                    q = thisq;
                    n0 = q[1] - p[1];
                    n1 = -q[0] + p[0];
                }
            }
        }

        // we must have elected *some* line, so long as there are at
        // least 2 points in the polygon.
        assert(q != NULL);

        // loop completed?
        if (q == pleft)
            break;

        int colinear = 0;

        // is this new point colinear with the last two?
        if (zarray_size(hull) > 1) {
            double *o;
            zarray_get_volatile(hull, zarray_size(hull) - 2, &o);

            double e0 = o[0] - p[0];
            double e1 = o[1] - p[1];

            if (n0*e0 + n1*e1 == 0)
                colinear = 1;
        }

        // if it is colinear, overwrite the last one.
        if (colinear)
            zarray_set(hull, zarray_size(hull)-1, q, NULL);
        else
            zarray_add(hull, q);

        p = q;
    }

    return hull;
}

// Find point p on the boundary of poly that is closest to q.
void g2d_polygon_closest_boundary_point(const zarray_t *poly, const double q[2], double *p)
{
    int psz = zarray_size(poly);
    double min_dist = HUGE_VALF;

    for (int i = 0; i < psz; i++) {
        double *p0, *p1;

        zarray_get_volatile(poly, i, &p0);
        zarray_get_volatile(poly, (i+1) % psz, &p1);

        g2d_line_segment_t seg;
        g2d_line_segment_init_from_points(&seg, p0, p1);

        double thisp[2];
        g2d_line_segment_closest_point(&seg, q, thisp);

        double dist = g2d_distance(q, thisp);
        if (dist < min_dist) {
            memcpy(p, thisp, sizeof(double[2]));
            min_dist = dist;
        }
    }
}

int g2d_polygon_contains_point(const zarray_t *poly, double q[2])
{
    // use winding. If the point is inside the polygon, we'll wrap
    // around it (accumulating 6.28 radians). If we're outside the
    // polygon, we'll accumulate zero.
    int psz = zarray_size(poly);
    assert(psz > 0);

    int last_quadrant;
    int quad_acc = 0;

    for (int i = 0; i <= psz; i++) {
        double *p;

        zarray_get_volatile(poly, i % psz, &p);

        // p[0] < q[0]       p[1] < q[1]    quadrant
        //     0                 0              0
        //     0                 1              3
        //     1                 0              1
        //     1                 1              2

        // p[1] < q[1]       p[0] < q[0]    quadrant
        //     0                 0              0
        //     0                 1              1
        //     1                 0              3
        //     1                 1              2

        int quadrant;
        if (p[0] < q[0])
            quadrant = (p[1] < q[1]) ? 2 : 1;
        else
            quadrant = (p[1] < q[1]) ? 3 : 0;

        if (i > 0) {
            int dquadrant = quadrant - last_quadrant;

            // encourage a jump table by mapping to small positive integers.
            switch (dquadrant) {
                case -3:
                case 1:
                    quad_acc ++;
                    break;
                case -1:
                case 3:
                    quad_acc --;
                    break;
                case 0:
                    break;
                case -2:
                case 2:
                {
                    // get the previous point.
                    double *p0;
                    zarray_get_volatile(poly, i-1, &p0);

                    // Consider the points p0 and p (the points around the
                    //polygon that we are tracing) and the query point q.
                    //
                    // If we've moved diagonally across quadrants, we want
                    // to measure whether we have rotated +PI radians or
                    // -PI radians. We can test this by computing the dot
                    // product of vector (p0-q) with the vector
                    // perpendicular to vector (p-q)
                    double nx = p[1] - q[1];
                    double ny = -p[0] + q[0];

                    double dot = nx*(p0[0]-q[0]) + ny*(p0[1]-q[1]);
                    if (dot < 0)
                        quad_acc -= 2;
                    else
                        quad_acc += 2;

                    break;
                }
            }
        }

        last_quadrant = quadrant;
    }

    int v = (quad_acc >= 2) || (quad_acc <= -2);

    if (0 && v != g2d_polygon_contains_point_ref(poly, q)) {
        printf("FAILURE %d %d\n", v, quad_acc);
        exit(-1);
    }

    return v;
}

void g2d_line_init_from_points(g2d_line_t *line, const double p0[2], const double p1[2])
{
    line->p[0] = p0[0];
    line->p[1] = p0[1];
    line->u[0] = p1[0]-p0[0];
    line->u[1] = p1[1]-p0[1];
    double mag = sqrtf(sq(line->u[0]) + sq(line->u[1]));

    line->u[0] /= mag;
    line->u[1] /= mag;
}

double g2d_line_get_coordinate(const g2d_line_t *line, const double q[2])
{
    return (q[0]-line->p[0])*line->u[0] + (q[1]-line->p[1])*line->u[1];
}

// Compute intersection of two line segments. If they intersect,
// result is stored in p and 1 is returned. Otherwise, zero is
// returned. p may be NULL.
int g2d_line_intersect_line(const g2d_line_t *linea, const g2d_line_t *lineb, double *p)
{
    // this implementation is many times faster than the original,
    // mostly due to avoiding a general-purpose LU decomposition in
    // Matrix.inverse().
    double m00, m01, m10, m11;
    double i00, i01;
    double b00, b10;

    m00 = linea->u[0];
    m01= -lineb->u[0];
    m10 = linea->u[1];
    m11= -lineb->u[1];

    // determinant of m
    double det = m00*m11-m01*m10;

    // parallel lines?
    if (fabs(det) < 0.00000001)
        return 0;

    // inverse of m
    i00 = m11/det;
    i01 = -m01/det;

    b00 = lineb->p[0] - linea->p[0];
    b10 = lineb->p[1] - linea->p[1];

    double x00; //, x10;
    x00 = i00*b00+i01*b10;

    if (p != NULL) {
        p[0] = linea->u[0]*x00 + linea->p[0];
        p[1] = linea->u[1]*x00 + linea->p[1];
    }

    return 1;
}


void g2d_line_segment_init_from_points(g2d_line_segment_t *seg, const double p0[2], const double p1[2])
{
    g2d_line_init_from_points(&seg->line, p0, p1);
    seg->p1[0] = p1[0];
    seg->p1[1] = p1[1];
}

// Find the point p on segment seg that is closest to point q.
void g2d_line_segment_closest_point(const g2d_line_segment_t *seg, const double *q, double *p)
{
    double a = g2d_line_get_coordinate(&seg->line, seg->line.p);
    double b = g2d_line_get_coordinate(&seg->line, seg->p1);
    double c = g2d_line_get_coordinate(&seg->line, q);

    if (a < b)
        c = dclamp(c, a, b);
    else
        c = dclamp(c, b, a);

    p[0] = seg->line.p[0] + c * seg->line.u[0];
    p[1] = seg->line.p[1] + c * seg->line.u[1];
}

// Compute intersection of two line segments. If they intersect,
// result is stored in p and 1 is returned. Otherwise, zero is
// returned. p may be NULL.
int g2d_line_segment_intersect_segment(const g2d_line_segment_t *sega, const g2d_line_segment_t *segb, double *p)
{
    double tmp[2];

    if (!g2d_line_intersect_line(&sega->line, &segb->line, tmp))
        return 0;

    double a = g2d_line_get_coordinate(&sega->line, sega->line.p);
    double b = g2d_line_get_coordinate(&sega->line, sega->p1);
    double c = g2d_line_get_coordinate(&sega->line, tmp);

    // does intersection lie on the first line?
    if ((c<a && c<b) || (c>a && c>b))
        return 0;

    a = g2d_line_get_coordinate(&segb->line, segb->line.p);
    b = g2d_line_get_coordinate(&segb->line, segb->p1);
    c = g2d_line_get_coordinate(&segb->line, tmp);

    // does intersection lie on second line?
    if ((c<a && c<b) || (c>a && c>b))
        return 0;

    if (p != NULL) {
        p[0] = tmp[0];
        p[1] = tmp[1];
    }

    return 1;
}

// Compute intersection of a line segment and a line. If they
// intersect, result is stored in p and 1 is returned. Otherwise, zero
// is returned. p may be NULL.
int g2d_line_segment_intersect_line(const g2d_line_segment_t *seg, const g2d_line_t *line, double *p)
{
    double tmp[2];

    if (!g2d_line_intersect_line(&seg->line, line, tmp))
        return 0;

    double a = g2d_line_get_coordinate(&seg->line, seg->line.p);
    double b = g2d_line_get_coordinate(&seg->line, seg->p1);
    double c = g2d_line_get_coordinate(&seg->line, tmp);

    // does intersection lie on the first line?
    if ((c<a && c<b) || (c>a && c>b))
        return 0;

    if (p != NULL) {
        p[0] = tmp[0];
        p[1] = tmp[1];
    }

    return 1;
}

// do the edges of polya and polyb collide? (Does NOT test for containment).
int g2d_polygon_intersects_polygon(const zarray_t *polya, const zarray_t *polyb)
{
    // do any of the line segments collide? If so, the answer is no.

    // dumb N^2 method.
    for (int ia = 0; ia < zarray_size(polya); ia++) {
        double pa0[2], pa1[2];
        zarray_get(polya, ia, pa0);
        zarray_get(polya, (ia+1)%zarray_size(polya), pa1);

        g2d_line_segment_t sega;
        g2d_line_segment_init_from_points(&sega, pa0, pa1);

        for (int ib = 0; ib < zarray_size(polyb); ib++) {
            double pb0[2], pb1[2];
            zarray_get(polyb, ib, pb0);
            zarray_get(polyb, (ib+1)%zarray_size(polyb), pb1);

            g2d_line_segment_t segb;
            g2d_line_segment_init_from_points(&segb, pb0, pb1);

            if (g2d_line_segment_intersect_segment(&sega, &segb, NULL))
                return 1;
        }
    }

    return 0;
}

// does polya completely contain polyb?
int g2d_polygon_contains_polygon(const zarray_t *polya, const zarray_t *polyb)
{
    // do any of the line segments collide? If so, the answer is no.
    if (g2d_polygon_intersects_polygon(polya, polyb))
        return 0;

    // if none of the edges cross, then the polygon is either fully
    // contained or fully outside.
    double p[2];
    zarray_get(polyb, 0, p);

    return g2d_polygon_contains_point(polya, p);
}

// compute a point that is inside the polygon. (It may not be *far* inside though)
void g2d_polygon_get_interior_point(const zarray_t *poly, double *p)
{
    // take the first three points, which form a triangle. Find the middle point
    double a[2], b[2], c[2];

    zarray_get(poly, 0, a);
    zarray_get(poly, 1, b);
    zarray_get(poly, 2, c);

    p[0] = (a[0]+b[0]+c[0])/3;
    p[1] = (a[1]+b[1]+c[1])/3;
}

int g2d_polygon_overlaps_polygon(const zarray_t *polya, const zarray_t *polyb)
{
    // do any of the line segments collide? If so, the answer is yes.
    if (g2d_polygon_intersects_polygon(polya, polyb))
        return 1;

    // if none of the edges cross, then the polygon is either fully
    // contained or fully outside.
    double p[2];
    g2d_polygon_get_interior_point(polyb, p);

    if (g2d_polygon_contains_point(polya, p))
        return 1;

    g2d_polygon_get_interior_point(polya, p);

    if (g2d_polygon_contains_point(polyb, p))
        return 1;

    return 0;
}

static int double_sort_up(const void *_a, const void *_b)
{
    double a = *((double*) _a);
    double b = *((double*) _b);

    if (a < b)
        return -1;

    if (a == b)
        return 0;

    return 1;
}

// Compute the crossings of the polygon along line y, storing them in
// the array x. X must be allocated to be at least as long as
// zarray_size(poly). X will be sorted, ready for
// rasterization. Returns the number of intersections (and elements
// written to x).
/*
  To rasterize, do something like this:

  double res = 0.099;
  for (double y = y0; y < y1; y += res) {
  double xs[zarray_size(poly)];

  int xsz = g2d_polygon_rasterize(poly, y, xs);
  int xpos = 0;
  int inout = 0; // start off "out"

  for (double x = x0; x < x1; x += res) {
      while (x > xs[xpos] && xpos < xsz) {
        xpos++;
        inout ^= 1;
      }

    if (inout)
       printf("y");
    else
       printf(" ");
  }
  printf("\n");
*/

// returns the number of x intercepts
int g2d_polygon_rasterize(const zarray_t *poly, double y, double *x)
{
    int sz = zarray_size(poly);

    g2d_line_t line;
    if (1) {
        double p0[2] = { 0, y };
        double p1[2] = { 1, y };

        g2d_line_init_from_points(&line, p0, p1);
    }

    int xpos = 0;

    for (int i = 0; i < sz; i++) {
        g2d_line_segment_t seg;
        double *p0, *p1;
        zarray_get_volatile(poly, i, &p0);
        zarray_get_volatile(poly, (i+1)%sz, &p1);

        g2d_line_segment_init_from_points(&seg, p0, p1);

        double q[2];
        if (g2d_line_segment_intersect_line(&seg, &line, q))
            x[xpos++] = q[0];
    }

    qsort(x, xpos, sizeof(double), double_sort_up);

    return xpos;
}

/*
  /---(1,5)
  (-2,4)-/        |
  \          |
  \        (1,2)--(2,2)\
  \                     \
  \                      \
  (0,0)------------------(4,0)
*/
#if 0

#include "timeprofile.h"

int main(int argc, char *argv[])
{
    timeprofile_t *tp = timeprofile_create();

    zarray_t *polya = g2d_polygon_create_data((double[][2]) {
            { 0, 0},
            { 4, 0},
            { 2, 2},
            { 1, 2},
            { 1, 5},
            { -2,4} }, 6);

    zarray_t *polyb = g2d_polygon_create_data((double[][2]) {
            { .1, .1},
            { .5, .1},
            { .1, .5 } }, 3);

    zarray_t *polyc = g2d_polygon_create_data((double[][2]) {
            { 3, 0},
            { 5, 0},
            { 5, 1} }, 3);

    zarray_t *polyd = g2d_polygon_create_data((double[][2]) {
            { 5, 5},
            { 6, 6},
            { 5, 6} }, 3);

/*
  5      L---K
  4      |I--J
  3      |H-G
  2      |E-F
  1      |D--C
  0      A---B
  01234
*/
    zarray_t *polyE = g2d_polygon_create_data((double[][2]) {
            {0,0}, {4,0}, {4, 1}, {1,1},
                                  {1,2}, {3,2}, {3,3}, {1,3},
                                                       {1,4}, {4,4}, {4,5}, {0,5}}, 12);

    srand(0);

    timeprofile_stamp(tp, "begin");

    if (1) {
        int niters = 100000;

        for (int i = 0; i < niters; i++) {
            double q[2];
            q[0] = 10.0f * random() / RAND_MAX - 2;
            q[1] = 10.0f * random() / RAND_MAX - 2;

            g2d_polygon_contains_point(polyE, q);
        }

        timeprofile_stamp(tp, "fast");

        for (int i = 0; i < niters; i++) {
            double q[2];
            q[0] = 10.0f * random() / RAND_MAX - 2;
            q[1] = 10.0f * random() / RAND_MAX - 2;

            g2d_polygon_contains_point_ref(polyE, q);
        }

        timeprofile_stamp(tp, "slow");

        for (int i = 0; i < niters; i++) {
            double q[2];
            q[0] = 10.0f * random() / RAND_MAX - 2;
            q[1] = 10.0f * random() / RAND_MAX - 2;

            int v0 = g2d_polygon_contains_point(polyE, q);
            int v1 = g2d_polygon_contains_point_ref(polyE, q);
            assert(v0 == v1);
        }

        timeprofile_stamp(tp, "both");
        timeprofile_display(tp);
    }

    if (1) {
        zarray_t *poly = polyE;

        double res = 0.399;
        for (double y = 5.2; y >= -.5; y -= res) {
            double xs[zarray_size(poly)];

            int xsz = g2d_polygon_rasterize(poly, y, xs);
            int xpos = 0;
            int inout = 0; // start off "out"
            for (double x = -3; x < 6; x += res) {
                while (x > xs[xpos] && xpos < xsz) {
                    xpos++;
                    inout ^= 1;
                }

                if (inout)
                    printf("y");
                else
                    printf(" ");
            }
            printf("\n");

            for (double x = -3; x < 6; x += res) {
                double q[2] = {x, y};
                if (g2d_polygon_contains_point(poly, q))
                    printf("X");
                else
                    printf(" ");
            }
            printf("\n");
        }
    }



/*
// CW order
double p[][2] =  { { 0, 0},
{ -2, 4},
{1, 5},
{1, 2},
{2, 2},
{4, 0} };
*/

     double q[2] = { 10, 10 };
     printf("0==%d\n", g2d_polygon_contains_point(polya, q));

     q[0] = 1; q[1] = 1;
     printf("1==%d\n", g2d_polygon_contains_point(polya, q));

     q[0] = 3; q[1] = .5;
     printf("1==%d\n", g2d_polygon_contains_point(polya, q));

     q[0] = 1.2; q[1] = 2.1;
     printf("0==%d\n", g2d_polygon_contains_point(polya, q));

     printf("0==%d\n", g2d_polygon_contains_polygon(polya, polyb));

     printf("0==%d\n", g2d_polygon_contains_polygon(polya, polyc));

     printf("0==%d\n", g2d_polygon_contains_polygon(polya, polyd));

     ////////////////////////////////////////////////////////
     // Test convex hull
     if (1) {
         zarray_t *hull = g2d_convex_hull(polyE);

         for (int k = 0; k < zarray_size(hull); k++) {
             double *h;
             zarray_get_volatile(hull, k, &h);

             printf("%15f, %15f\n", h[0], h[1]);
         }
     }

     for (int i = 0; i < 100000; i++) {
         zarray_t *points = zarray_create(sizeof(double[2]));

         for (int j = 0; j < 100; j++) {
             double q[2];
             q[0] = 10.0f * random() / RAND_MAX - 2;
             q[1] = 10.0f * random() / RAND_MAX - 2;

             zarray_add(points, q);
         }

         zarray_t *hull = g2d_convex_hull(points);
         for (int j = 0; j < zarray_size(points); j++) {
             double *q;
             zarray_get_volatile(points, j, &q);

             int on_edge;

             double p[2];
             g2d_polygon_closest_boundary_point(hull, q, p);
             if (g2d_distance(q, p) < .00001)
                 on_edge = 1;

             assert(on_edge || g2d_polygon_contains_point(hull, q));
         }

         zarray_destroy(hull);
         zarray_destroy(points);
     }
}
#endif
