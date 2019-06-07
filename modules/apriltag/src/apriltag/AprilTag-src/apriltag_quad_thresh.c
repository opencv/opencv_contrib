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

// limitation: image size must be <32768 in width and height. This is
// because we use a fixed-point 16 bit integer representation with one
// fractional bit.
#include <math.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

#include "apriltag.h"
#include "common/image_u8x3.h"
#include "common/zarray.h"
#include "common/zhash.h"
#include "common/unionfind.h"
#include "common/timeprofile.h"
#include "common/zmaxheap.h"
#include "common/postscript_utils.h"
#include "common/math_util.h"

#ifdef _WIN32
static inline long int random(void)
{
        return rand();
}
#endif

static inline uint32_t u64hash_2(uint64_t x) {
    return (2654435761 * x) >> 32;
}

struct uint64_zarray_entry
{
    uint64_t id;
    zarray_t *cluster;

    struct uint64_zarray_entry *next;
};

#ifndef M_PI
# define M_PI 3.141592653589793238462643383279502884196
#endif

struct pt
{
    // Note: these represent 2*actual value.
    uint16_t x, y;
    int16_t gx, gy;

    float slope;
};

struct unionfind_task
{
    int y0, y1;
    int w, h, s;
    unionfind_t *uf;
    image_u8_t *im;
};

struct quad_task
{
    zarray_t *clusters;
    int cidx0, cidx1; // [cidx0, cidx1)
    zarray_t *quads;
    apriltag_detector_t *td;
    int w, h;

    image_u8_t *im;
    int tag_width;
    bool normal_border;
    bool reversed_border;
};


struct cluster_task
{
    int y0;
    int y1;
    int w;
    int s;
    int nclustermap;
    unionfind_t* uf;
    image_u8_t* im;
    zarray_t* clusters;
};

struct remove_vertex
{
    int i;           // which vertex to remove?
    int left, right; // left vertex, right vertex

    double err;
};

struct segment
{
    int is_vertex;

    // always greater than zero, but right can be > size, which denotes
    // a wrap around back to the beginning of the points. and left < right.
    int left, right;
};

struct line_fit_pt
{
    double Mx, My;
    double Mxx, Myy, Mxy;
    double W; // total weight
};

struct cluster_hash
{
    uint32_t hash;
    uint64_t id;
    zarray_t* data;
};


// lfps contains *cumulative* moments for N points, with
// index j reflecting points [0,j] (inclusive).
//
// fit a line to the points [i0, i1] (inclusive). i0, i1 are both [0,
// sz) if i1 < i0, we treat this as a wrap around.
void fit_line(struct line_fit_pt *lfps, int sz, int i0, int i1, double *lineparm, double *err, double *mse)
{
    assert(i0 != i1);
    assert(i0 >= 0 && i1 >= 0 && i0 < sz && i1 < sz);

    double Mx, My, Mxx, Myy, Mxy, W;
    int N; // how many points are included in the set?

    if (i0 < i1) {
        N = i1 - i0 + 1;

        Mx  = lfps[i1].Mx;
        My  = lfps[i1].My;
        Mxx = lfps[i1].Mxx;
        Mxy = lfps[i1].Mxy;
        Myy = lfps[i1].Myy;
        W   = lfps[i1].W;

        if (i0 > 0) {
            Mx  -= lfps[i0-1].Mx;
            My  -= lfps[i0-1].My;
            Mxx -= lfps[i0-1].Mxx;
            Mxy -= lfps[i0-1].Mxy;
            Myy -= lfps[i0-1].Myy;
            W   -= lfps[i0-1].W;
        }

    } else {
        // i0 > i1, e.g. [15, 2]. Wrap around.
        assert(i0 > 0);

        Mx  = lfps[sz-1].Mx   - lfps[i0-1].Mx;
        My  = lfps[sz-1].My   - lfps[i0-1].My;
        Mxx = lfps[sz-1].Mxx  - lfps[i0-1].Mxx;
        Mxy = lfps[sz-1].Mxy  - lfps[i0-1].Mxy;
        Myy = lfps[sz-1].Myy  - lfps[i0-1].Myy;
        W   = lfps[sz-1].W    - lfps[i0-1].W;

        Mx  += lfps[i1].Mx;
        My  += lfps[i1].My;
        Mxx += lfps[i1].Mxx;
        Mxy += lfps[i1].Mxy;
        Myy += lfps[i1].Myy;
        W   += lfps[i1].W;

        N = sz - i0 + i1 + 1;
    }

    assert(N >= 2);

    double Ex = Mx / W;
    double Ey = My / W;
    double Cxx = Mxx / W - Ex*Ex;
    double Cxy = Mxy / W - Ex*Ey;
    double Cyy = Myy / W - Ey*Ey;

    //if (1) {
    //    // on iOS about 5% of total CPU spent in these trig functions.
    //    // 85 ms per frame on 5S, example.pnm
    //    //
    //    // XXX this was using the double-precision atan2. Was there a case where
    //    // we needed that precision? Seems doubtful.
    //    double normal_theta = .5 * atan2f(-2*Cxy, (Cyy - Cxx));
    //    nx_old = cosf(normal_theta);
    //    ny_old = sinf(normal_theta);
    //}

    // Instead of using the above cos/sin method, pose it as an eigenvalue problem.
    double eig_small = 0.5*(Cxx + Cyy - sqrtf((Cxx - Cyy)*(Cxx - Cyy) + 4*Cxy*Cxy));

    if (lineparm) {
        lineparm[0] = Ex;
        lineparm[1] = Ey;

        double eig = 0.5*(Cxx + Cyy + sqrtf((Cxx - Cyy)*(Cxx - Cyy) + 4*Cxy*Cxy));
        double nx1 = Cxx - eig;
        double ny1 = Cxy;
        double M1 = nx1*nx1 + ny1*ny1;
        double nx2 = Cxy;
        double ny2 = Cyy - eig;
        double M2 = nx2*nx2 + ny2*ny2;

        double nx, ny, M;
        if (M1 > M2) {
            nx = nx1;
            ny = ny1;
            M = M1;
        } else {
            nx = nx2;
            ny = ny2;
            M = M2;
        }

        double length = sqrtf(M);
        lineparm[2] = nx/length;
        lineparm[3] = ny/length;
    }

    // sum of squared errors =
    //
    // SUM_i ((p_x - ux)*nx + (p_y - uy)*ny)^2
    // SUM_i  nx*nx*(p_x - ux)^2 + 2nx*ny(p_x -ux)(p_y-uy) + ny*ny*(p_y-uy)*(p_y-uy)
    //  nx*nx*SUM_i((p_x -ux)^2) + 2nx*ny*SUM_i((p_x-ux)(p_y-uy)) + ny*ny*SUM_i((p_y-uy)^2)
    //
    //  nx*nx*N*Cxx + 2nx*ny*N*Cxy + ny*ny*N*Cyy

    // sum of squared errors
    if (err)
        *err = N*eig_small;

    // mean squared error
    if (mse)
        *mse = eig_small;
}

float pt_compare_angle(struct pt *a, struct pt *b) {
    return a->slope - b->slope;
}

int err_compare_descending(const void *_a, const void *_b)
{
    const double *a =  _a;
    const double *b =  _b;

    return ((*a) < (*b)) ? 1 : -1;
}

/*

  1. Identify A) white points near a black point and B) black points near a white point.

  2. Find the connected components within each of the classes above,
  yielding clusters of "white-near-black" and
  "black-near-white". (These two classes are kept separate). Each
  segment has a unique id.

  3. For every pair of "white-near-black" and "black-near-white"
  clusters, find the set of points that are in one and adjacent to the
  other. In other words, a "boundary" layer between the two
  clusters. (This is actually performed by iterating over the pixels,
  rather than pairs of clusters.) Critically, this helps keep nearby
  edges from becoming connected.
*/
int quad_segment_maxima(apriltag_detector_t *td, zarray_t *cluster, struct line_fit_pt *lfps, int indices[4])
{
    int sz = zarray_size(cluster);

    // ksz: when fitting points, how many points on either side do we consider?
    // (actual "kernel" width is 2ksz).
    //
    // This value should be about: 0.5 * (points along shortest edge).
    //
    // If all edges were equally-sized, that would give a value of
    // sz/8. We make it somewhat smaller to account for tags at high
    // aspects.

    // XXX Tunable. Maybe make a multiple of JPEG block size to increase robustness
    // to JPEG compression artifacts?
    int ksz = imin(20, sz / 12);

    // can't fit a quad if there are too few points.
    if (ksz < 2)
        return 0;

    double *errs = malloc(sizeof(double)*sz);

    for (int i = 0; i < sz; i++) {
        fit_line(lfps, sz, (i + sz - ksz) % sz, (i + ksz) % sz, NULL, &errs[i], NULL);
    }

    // apply a low-pass filter to errs
    if (1) {
        double *y = malloc(sizeof(double)*sz);

        // how much filter to apply?

        // XXX Tunable
        double sigma = 1; // was 3

        // cutoff = exp(-j*j/(2*sigma*sigma));
        // log(cutoff) = -j*j / (2*sigma*sigma)
        // log(cutoff)*2*sigma*sigma = -j*j;

        // how big a filter should we use? We make our kernel big
        // enough such that we represent any values larger than
        // 'cutoff'.

        // XXX Tunable (though not super useful to change)
        double cutoff = 0.05;
        int fsz = sqrt(-log(cutoff)*2*sigma*sigma) + 1;
        fsz = 2*fsz + 1;

        // For default values of cutoff = 0.05, sigma = 3,
        // we have fsz = 17.
        float *f = malloc(sizeof(float)*fsz);

        for (int i = 0; i < fsz; i++) {
            int j = i - fsz / 2;
            f[i] = exp(-j*j/(2*sigma*sigma));
        }

        for (int iy = 0; iy < sz; iy++) {
            double acc = 0;

            for (int i = 0; i < fsz; i++) {
                acc += errs[(iy + i - fsz / 2 + sz) % sz] * f[i];
            }
            y[iy] = acc;
        }

        memcpy(errs, y, sizeof(double)*sz);
        free(y);
        free(f);
    }

    int *maxima = malloc(sizeof(int)*sz);
    double *maxima_errs = malloc(sizeof(double)*sz);
    int nmaxima = 0;

    for (int i = 0; i < sz; i++) {
        if (errs[i] > errs[(i+1)%sz] && errs[i] > errs[(i+sz-1)%sz]) {
            maxima[nmaxima] = i;
            maxima_errs[nmaxima] = errs[i];
            nmaxima++;
        }
    }
    free(errs);

    // if we didn't get at least 4 maxima, we can't fit a quad.
    if (nmaxima < 4)
        return 0;

    // select only the best maxima if we have too many
    int max_nmaxima = td->qtp.max_nmaxima;

    if (nmaxima > max_nmaxima) {
        double *maxima_errs_copy = malloc(sizeof(double)*nmaxima);
        memcpy(maxima_errs_copy, maxima_errs, sizeof(double)*nmaxima);

        // throw out all but the best handful of maxima. Sorts descending.
        qsort(maxima_errs_copy, nmaxima, sizeof(double), err_compare_descending);

        double maxima_thresh = maxima_errs_copy[max_nmaxima];
        int out = 0;
        for (int in = 0; in < nmaxima; in++) {
            if (maxima_errs[in] <= maxima_thresh)
                continue;
            maxima[out++] = maxima[in];
        }
        nmaxima = out;
        free(maxima_errs_copy);
    }
    free(maxima_errs);

    int best_indices[4];
    double best_error = HUGE_VALF;

    double err01, err12, err23, err30;
    double mse01, mse12, mse23, mse30;
    double params01[4], params12[4], params23[4], params30[4];

    // disallow quads where the angle is less than a critical value.
    double max_dot = td->qtp.cos_critical_rad; //25*M_PI/180);

    for (int m0 = 0; m0 < nmaxima - 3; m0++) {
        int i0 = maxima[m0];

        for (int m1 = m0+1; m1 < nmaxima - 2; m1++) {
            int i1 = maxima[m1];

            fit_line(lfps, sz, i0, i1, params01, &err01, &mse01);

            if (mse01 > td->qtp.max_line_fit_mse)
                continue;

            for (int m2 = m1+1; m2 < nmaxima - 1; m2++) {
                int i2 = maxima[m2];

                fit_line(lfps, sz, i1, i2, params12, &err12, &mse12);
                if (mse12 > td->qtp.max_line_fit_mse)
                    continue;

                double dot = params01[2]*params12[2] + params01[3]*params12[3];
                if (fabs(dot) > max_dot)
                    continue;

                for (int m3 = m2+1; m3 < nmaxima; m3++) {
                    int i3 = maxima[m3];

                    fit_line(lfps, sz, i2, i3, params23, &err23, &mse23);
                    if (mse23 > td->qtp.max_line_fit_mse)
                        continue;

                    fit_line(lfps, sz, i3, i0, params30, &err30, &mse30);
                    if (mse30 > td->qtp.max_line_fit_mse)
                        continue;

                    double err = err01 + err12 + err23 + err30;
                    if (err < best_error) {
                        best_error = err;
                        best_indices[0] = i0;
                        best_indices[1] = i1;
                        best_indices[2] = i2;
                        best_indices[3] = i3;
                    }
                }
            }
        }
    }

    free(maxima);

    if (best_error == HUGE_VALF)
        return 0;

    for (int i = 0; i < 4; i++)
        indices[i] = best_indices[i];

    if (best_error / sz < td->qtp.max_line_fit_mse)
        return 1;
    return 0;
}

// returns 0 if the cluster looks bad.
int quad_segment_agg(apriltag_detector_t *td, zarray_t *cluster, struct line_fit_pt *lfps, int indices[4])
{
    int sz = zarray_size(cluster);

    zmaxheap_t *heap = zmaxheap_create(sizeof(struct remove_vertex*));

    // We will initially allocate sz rvs. We then have two types of
    // iterations: some iterations that are no-ops in terms of
    // allocations, and those that remove a vertex and allocate two
    // more children.  This will happen at most (sz-4) times.  Thus we
    // need: sz + 2*(sz-4) entries.

    int rvalloc_pos = 0;
    int rvalloc_size = 3*sz;
    struct remove_vertex *rvalloc = calloc(rvalloc_size, sizeof(struct remove_vertex));

    struct segment *segs = calloc(sz, sizeof(struct segment));

    // populate with initial entries
    for (int i = 0; i < sz; i++) {
        struct remove_vertex *rv = &rvalloc[rvalloc_pos++];
        rv->i = i;
        if (i == 0) {
            rv->left = sz-1;
            rv->right = 1;
        } else {
            rv->left  = i-1;
            rv->right = (i+1) % sz;
        }

        fit_line(lfps, sz, rv->left, rv->right, NULL, NULL, &rv->err);

        zmaxheap_add(heap, &rv, -rv->err);

        segs[i].left = rv->left;
        segs[i].right = rv->right;
        segs[i].is_vertex = 1;
    }

    int nvertices = sz;

    while (nvertices > 4) {
        assert(rvalloc_pos < rvalloc_size);

        struct remove_vertex *rv;
        float err;

        int res = zmaxheap_remove_max(heap, &rv, &err);
        if (!res)
            return 0;
        assert(res);

        // is this remove_vertex valid? (Or has one of the left/right
        // vertices changes since we last looked?)
        if (!segs[rv->i].is_vertex ||
            !segs[rv->left].is_vertex ||
            !segs[rv->right].is_vertex) {
            continue;
        }

        // we now merge.
        assert(segs[rv->i].is_vertex);

        segs[rv->i].is_vertex = 0;
        segs[rv->left].right = rv->right;
        segs[rv->right].left = rv->left;

        // create the join to the left
        if (1) {
            struct remove_vertex *child = &rvalloc[rvalloc_pos++];
            child->i = rv->left;
            child->left = segs[rv->left].left;
            child->right = rv->right;

            fit_line(lfps, sz, child->left, child->right, NULL, NULL, &child->err);

            zmaxheap_add(heap, &child, -child->err);
        }

        // create the join to the right
        if (1) {
            struct remove_vertex *child = &rvalloc[rvalloc_pos++];
            child->i = rv->right;
            child->left = rv->left;
            child->right = segs[rv->right].right;

            fit_line(lfps, sz, child->left, child->right, NULL, NULL, &child->err);

            zmaxheap_add(heap, &child, -child->err);
        }

        // we now have one less vertex
        nvertices--;
    }

    free(rvalloc);
    zmaxheap_destroy(heap);

    int idx = 0;
    for (int i = 0; i < sz; i++) {
        if (segs[i].is_vertex) {
            indices[idx++] = i;
        }
    }

    free(segs);

    return 1;
}

/**
 * Compute statistics that allow line fit queries to be
 * efficiently computed for any contiguous range of indices.
 */
struct line_fit_pt* compute_lfps(int sz, zarray_t* cluster, image_u8_t* im) {
    struct line_fit_pt *lfps = calloc(sz, sizeof(struct line_fit_pt));

    for (int i = 0; i < sz; i++) {
        struct pt *p;
        zarray_get_volatile(cluster, i, &p);

        if (i > 0) {
            memcpy(&lfps[i], &lfps[i-1], sizeof(struct line_fit_pt));
        }

        {
            // we now undo our fixed-point arithmetic.
            double delta = 0.5; // adjust for pixel center bias
            double x = p->x * .5 + delta;
            double y = p->y * .5 + delta;
            int ix = x, iy = y;
            double W = 1;

            if (ix > 0 && ix+1 < im->width && iy > 0 && iy+1 < im->height) {
                int grad_x = im->buf[iy * im->stride + ix + 1] -
                    im->buf[iy * im->stride + ix - 1];

                int grad_y = im->buf[(iy+1) * im->stride + ix] -
                    im->buf[(iy-1) * im->stride + ix];

                // XXX Tunable. How to shape the gradient magnitude?
                W = sqrt(grad_x*grad_x + grad_y*grad_y) + 1;
            }

            double fx = x, fy = y;
            lfps[i].Mx  += W * fx;
            lfps[i].My  += W * fy;
            lfps[i].Mxx += W * fx * fx;
            lfps[i].Mxy += W * fx * fy;
            lfps[i].Myy += W * fy * fy;
            lfps[i].W   += W;
        }
    }
    return lfps;
}

static inline void ptsort(struct pt *pts, int sz)
{
#define MAYBE_SWAP(arr,apos,bpos)                                   \
    if (pt_compare_angle(&(arr[apos]), &(arr[bpos])) > 0) {                        \
        tmp = arr[apos]; arr[apos] = arr[bpos]; arr[bpos] = tmp;    \
    };

    if (sz <= 1)
        return;

    if (sz == 2) {
        struct pt tmp;
        MAYBE_SWAP(pts, 0, 1);
        return;
    }

    // NB: Using less-branch-intensive sorting networks here on the
    // hunch that it's better for performance.
    if (sz == 3) { // 3 element bubble sort is optimal
        struct pt tmp;
        MAYBE_SWAP(pts, 0, 1);
        MAYBE_SWAP(pts, 1, 2);
        MAYBE_SWAP(pts, 0, 1);
        return;
    }

    if (sz == 4) { // 4 element optimal sorting network.
        struct pt tmp;
        MAYBE_SWAP(pts, 0, 1); // sort each half, like a merge sort
        MAYBE_SWAP(pts, 2, 3);
        MAYBE_SWAP(pts, 0, 2); // minimum value is now at 0.
        MAYBE_SWAP(pts, 1, 3); // maximum value is now at end.
        MAYBE_SWAP(pts, 1, 2); // that only leaves the middle two.
        return;
    }
    if (sz == 5) {
        // this 9-step swap is optimal for a sorting network, but two
        // steps slower than a generic sort.
        struct pt tmp;
        MAYBE_SWAP(pts, 0, 1); // sort each half (3+2), like a merge sort
        MAYBE_SWAP(pts, 3, 4);
        MAYBE_SWAP(pts, 1, 2);
        MAYBE_SWAP(pts, 0, 1);
        MAYBE_SWAP(pts, 0, 3); // minimum element now at 0
        MAYBE_SWAP(pts, 2, 4); // maximum element now at end
        MAYBE_SWAP(pts, 1, 2); // now resort the three elements 1-3.
        MAYBE_SWAP(pts, 2, 3);
        MAYBE_SWAP(pts, 1, 2);
        return;
    }

#undef MAYBE_SWAP

    // a merge sort with temp storage.

    struct pt *tmp = malloc(sizeof(struct pt) * sz);

    memcpy(tmp, pts, sizeof(struct pt) * sz);

    int asz = sz/2;
    int bsz = sz - asz;

    struct pt *as = &tmp[0];
    struct pt *bs = &tmp[asz];

    ptsort(as, asz);
    ptsort(bs, bsz);

    #define MERGE(apos,bpos)                        \
    if (pt_compare_angle(&(as[apos]), &(bs[bpos])) < 0)        \
        pts[outpos++] = as[apos++];             \
    else                                        \
        pts[outpos++] = bs[bpos++];

    int apos = 0, bpos = 0, outpos = 0;
    while (apos + 8 < asz && bpos + 8 < bsz) {
        MERGE(apos,bpos); MERGE(apos,bpos); MERGE(apos,bpos); MERGE(apos,bpos);
        MERGE(apos,bpos); MERGE(apos,bpos); MERGE(apos,bpos); MERGE(apos,bpos);
    }

    while (apos < asz && bpos < bsz) {
        MERGE(apos,bpos);
    }

    if (apos < asz)
        memcpy(&pts[outpos], &as[apos], (asz-apos)*sizeof(struct pt));
    if (bpos < bsz)
        memcpy(&pts[outpos], &bs[bpos], (bsz-bpos)*sizeof(struct pt));

    free(tmp);

#undef MERGE
}

// return 1 if the quad looks okay, 0 if it should be discarded
int fit_quad(
        apriltag_detector_t *td,
        image_u8_t *im,
        zarray_t *cluster,
        struct quad *quad,
        int tag_width,
        bool normal_border,
        bool reversed_border) {
    int res = 0;

    int sz = zarray_size(cluster);
    if (sz < 24) // Synchronize with later check.
        return 0;

    /////////////////////////////////////////////////////////////
    // Step 1. Sort points so they wrap around the center of the
    // quad. We will constrain our quad fit to simply partition this
    // ordered set into 4 groups.

    // compute a bounding box so that we can order the points
    // according to their angle WRT the center.
    struct pt *p1;
    zarray_get_volatile(cluster, 0, &p1);
    uint16_t xmax = p1->x;
    uint16_t xmin = p1->x;
    uint16_t ymax = p1->y;
    uint16_t ymin = p1->y;
    for (int pidx = 1; pidx < zarray_size(cluster); pidx++) {
        struct pt *p;
        zarray_get_volatile(cluster, pidx, &p);

        if (p->x > xmax) {
            xmax = p->x;
        } else if (p->x < xmin) {
            xmin = p->x;
        }

        if (p->y > ymax) {
            ymax = p->y;
        } else if (p->y < ymin) {
            ymin = p->y;
        }
    }

    if ((xmax - xmin)*(ymax - ymin) < tag_width) {
        return 0;
    }

    // add some noise to (cx,cy) so that pixels get a more diverse set
    // of theta estimates. This will help us remove more points.
    // (Only helps a small amount. The actual noise values here don't
    // matter much at all, but we want them [-1, 1]. (XXX with
    // fixed-point, should range be bigger?)
    float cx = (xmin + xmax) * 0.5 + 0.05118;
    float cy = (ymin + ymax) * 0.5 + -0.028581;

    float dot = 0;

    float quadrants[2][2] = {{-1*(2 << 15), 0}, {2*(2 << 15), 2 << 15}};

    for (int pidx = 0; pidx < zarray_size(cluster); pidx++) {
        struct pt *p;
        zarray_get_volatile(cluster, pidx, &p);

        float dx = p->x - cx;
        float dy = p->y - cy;

        dot += dx*p->gx + dy*p->gy;

        float quadrant = quadrants[dy > 0][dx > 0];
        if (dy < 0) {
            dy = -dy;
            dx = -dx;
        }

        if (dx < 0) {
            float tmp = dx;
            dx = dy;
            dy = -tmp;
        }
        p->slope = quadrant + dy/dx;
    }

    // Ensure that the black border is inside the white border.
    quad->reversed_border = dot < 0;
    if (!reversed_border && quad->reversed_border) {
        return 0;
    }
    if (!normal_border && !quad->reversed_border) {
        return 0;
    }

    // we now sort the points according to theta. This is a prepatory
    // step for segmenting them into four lines.
    if (1) {
        ptsort((struct pt*) cluster->data, zarray_size(cluster));

        // remove duplicate points. (A byproduct of our segmentation system.)
        if (1) {
            int outpos = 1;

            struct pt *last;
            zarray_get_volatile(cluster, 0, &last);

            for (int i = 1; i < sz; i++) {

                struct pt *p;
                zarray_get_volatile(cluster, i, &p);

                if (p->x != last->x || p->y != last->y) {

                    if (i != outpos)  {
                        struct pt *out;
                        zarray_get_volatile(cluster, outpos, &out);
                        memcpy(out, p, sizeof(struct pt));
                    }

                    outpos++;
                }

                last = p;
            }

            cluster->size = outpos;
            sz = outpos;
        }

    }

    if (sz < 24)
        return 0;


    struct line_fit_pt *lfps = compute_lfps(sz, cluster, im);

    int indices[4];
    if (1) {
        if (!quad_segment_maxima(td, cluster, lfps, indices))
            goto finish;
    } else {
        if (!quad_segment_agg(td, cluster, lfps, indices))
            goto finish;
    }


    double lines[4][4];

    for (int i = 0; i < 4; i++) {
        int i0 = indices[i];
        int i1 = indices[(i+1)&3];

        double err;
        fit_line(lfps, sz, i0, i1, lines[i], NULL, &err);

        if (err > td->qtp.max_line_fit_mse) {
            res = 0;
            goto finish;
        }
    }

    for (int i = 0; i < 4; i++) {
        // solve for the intersection of lines (i) and (i+1)&3.
        // p0 + lambda0*u0 = p1 + lambda1*u1, where u0 and u1
        // are the line directions.
        //
        // lambda0*u0 - lambda1*u1 = (p1 - p0)
        //
        // rearrange (solve for lambdas)
        //
        // [u0_x   -u1_x ] [lambda0] = [ p1_x - p0_x ]
        // [u0_y   -u1_y ] [lambda1]   [ p1_y - p0_y ]
        //
        // remember that lines[i][0,1] = p, lines[i][2,3] = NORMAL vector.
        // We want the unit vector, so we need the perpendiculars. Thus, below
        // we have swapped the x and y components and flipped the y components.

        double A00 =  lines[i][3],  A01 = -lines[(i+1)&3][3];
        double A10 =  -lines[i][2],  A11 = lines[(i+1)&3][2];
        double B0 = -lines[i][0] + lines[(i+1)&3][0];
        double B1 = -lines[i][1] + lines[(i+1)&3][1];

        double det = A00 * A11 - A10 * A01;

        // inverse.
        double W00 = A11 / det, W01 = -A01 / det;
        if (fabs(det) < 0.001) {
            res = 0;
            goto finish;
        }

        // solve
        double L0 = W00*B0 + W01*B1;

        // compute intersection
        quad->p[i][0] = lines[i][0] + L0*A00;
        quad->p[i][1] = lines[i][1] + L0*A10;

        res = 1;
    }

    // reject quads that are too small
    if (1) {
        double area = 0;

        // get area of triangle formed by points 0, 1, 2, 0
        double length[3], p;
        for (int i = 0; i < 3; i++) {
            int idxa = i; // 0, 1, 2,
            int idxb = (i+1) % 3; // 1, 2, 0
            length[i] = sqrt(sq(quad->p[idxb][0] - quad->p[idxa][0]) +
                             sq(quad->p[idxb][1] - quad->p[idxa][1]));
        }
        p = (length[0] + length[1] + length[2]) / 2;

        area += sqrt(p*(p-length[0])*(p-length[1])*(p-length[2]));

        // get area of triangle formed by points 2, 3, 0, 2
        for (int i = 0; i < 3; i++) {
            int idxs[] = { 2, 3, 0, 2 };
            int idxa = idxs[i];
            int idxb = idxs[i+1];
            length[i] = sqrt(sq(quad->p[idxb][0] - quad->p[idxa][0]) +
                             sq(quad->p[idxb][1] - quad->p[idxa][1]));
        }
        p = (length[0] + length[1] + length[2]) / 2;

        area += sqrt(p*(p-length[0])*(p-length[1])*(p-length[2]));

        if (area < tag_width*tag_width) {
            res = 0;
            goto finish;
        }
    }

    // reject quads whose cumulative angle change isn't equal to 2PI
    if (1) {
        for (int i = 0; i < 4; i++) {
            int i0 = i, i1 = (i+1)&3, i2 = (i+2)&3;

            double dx1 = quad->p[i1][0] - quad->p[i0][0];
            double dy1 = quad->p[i1][1] - quad->p[i0][1];
            double dx2 = quad->p[i2][0] - quad->p[i1][0];
            double dy2 = quad->p[i2][1] - quad->p[i1][1];
            double cos_dtheta = (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2));

            if ((cos_dtheta > td->qtp.cos_critical_rad || cos_dtheta < -td->qtp.cos_critical_rad) || dx1*dy2 < dy1*dx2) {
                res = 0;
                goto finish;
            }
        }
    }

  finish:

    free(lfps);

    return res;
}

#define DO_UNIONFIND2(dx, dy) if (im->buf[(y + dy)*s + x + dx] == v) unionfind_connect(uf, y*w + x, (y + dy)*w + x + dx);

static void do_unionfind_first_line(unionfind_t *uf, image_u8_t *im, int h, int w, int s)
{
    int y = 0;
    uint8_t v;

    for (int x = 1; x < w - 1; x++) {
        v = im->buf[y*s + x];

        if (v == 127)
            continue;

        DO_UNIONFIND2(-1, 0);
    }
}

static void do_unionfind_line2(unionfind_t *uf, image_u8_t *im, int h, int w, int s, int y)
{
    assert(y > 0);

    uint8_t v_m1_m1;
    uint8_t v_0_m1 = im->buf[(y - 1)*s];
    uint8_t v_1_m1 = im->buf[(y - 1)*s + 1];
    uint8_t v_m1_0;
    uint8_t v = im->buf[y*s];

    for (int x = 1; x < w - 1; x++) {
        v_m1_m1 = v_0_m1;
        v_0_m1 = v_1_m1;
        v_1_m1 = im->buf[(y - 1)*s + x + 1];
        v_m1_0 = v;
        v = im->buf[y*s + x];

        if (v == 127)
            continue;

        // (dx,dy) pairs for 8 connectivity:
        // (-1, -1)    (0, -1)    (1, -1)
        // (-1, 0)    (REFERENCE)
        DO_UNIONFIND2(-1, 0);

        if (x == 1 || !((v_m1_0 == v_m1_m1) && (v_m1_m1 == v_0_m1))) {
            DO_UNIONFIND2(0, -1);
        }

        if (v == 255) {
            if (x == 1 || !(v_m1_0 == v_m1_m1 || v_0_m1 == v_m1_m1) ) {
                DO_UNIONFIND2(-1, -1);
            }
            if (!(v_0_m1 == v_1_m1)) {
                DO_UNIONFIND2(1, -1);
            }
        }
    }
}
#undef DO_UNIONFIND2

static void do_unionfind_task2(void *p)
{
    struct unionfind_task *task = (struct unionfind_task*) p;

    for (int y = task->y0; y < task->y1; y++) {
        do_unionfind_line2(task->uf, task->im, task->h, task->w, task->s, y);
    }
}

static void do_quad_task(void *p)
{
    struct quad_task *task = (struct quad_task*) p;

    zarray_t *clusters = task->clusters;
    zarray_t *quads = task->quads;
    apriltag_detector_t *td = task->td;
    int w = task->w, h = task->h;

    for (int cidx = task->cidx0; cidx < task->cidx1; cidx++) {

        zarray_t *cluster;
        zarray_get(clusters, cidx, &cluster);

        if (zarray_size(cluster) < td->qtp.min_cluster_pixels)
            continue;

        // a cluster should contain only boundary points around the
        // tag. it cannot be bigger than the whole screen. (Reject
        // large connected blobs that will be prohibitively slow to
        // fit quads to.) A typical point along an edge is added three
        // times (because it has 3 neighbors). The maximum perimeter
        // is 2w+2h.
        if (zarray_size(cluster) > 3*(2*w+2*h)) {
            continue;
        }

        struct quad quad;
        memset(&quad, 0, sizeof(struct quad));

        if (fit_quad(td, task->im, cluster, &quad, task->tag_width, task->normal_border, task->reversed_border)) {
            pthread_mutex_lock(&td->mutex);
            zarray_add(quads, &quad);
            pthread_mutex_unlock(&td->mutex);
        }
    }
}

image_u8_t *threshold(apriltag_detector_t *td, image_u8_t *im)
{
    int w = im->width, h = im->height, s = im->stride;
    assert(w < 32768);
    assert(h < 32768);

    image_u8_t *threshim = image_u8_create_alignment(w, h, s);
    assert(threshim->stride == s);

    // The idea is to find the maximum and minimum values in a
    // window around each pixel. If it's a contrast-free region
    // (max-min is small), don't try to binarize. Otherwise,
    // threshold according to (max+min)/2.
    //
    // Mark low-contrast regions with value 127 so that we can skip
    // future work on these areas too.

    // however, computing max/min around every pixel is needlessly
    // expensive. We compute max/min for tiles. To avoid artifacts
    // that arise when high-contrast features appear near a tile
    // edge (and thus moving from one tile to another results in a
    // large change in max/min value), the max/min values used for
    // any pixel are computed from all 3x3 surrounding tiles. Thus,
    // the max/min sampling area for nearby pixels overlap by at least
    // one tile.
    //
    // The important thing is that the windows be large enough to
    // capture edge transitions; the tag does not need to fit into
    // a tile.

    // XXX Tunable. Generally, small tile sizes--- so long as they're
    // large enough to span a single tag edge--- seem to be a winner.
    const int tilesz = 4;

    // the last (possibly partial) tiles along each row and column will
    // just use the min/max value from the last full tile.
    int tw = w / tilesz;
    int th = h / tilesz;

    uint8_t *im_max = calloc(tw*th, sizeof(uint8_t));
    uint8_t *im_min = calloc(tw*th, sizeof(uint8_t));

    // first, collect min/max statistics for each tile
    for (int ty = 0; ty < th; ty++) {
        for (int tx = 0; tx < tw; tx++) {
            uint8_t max = 0, min = 255;

            for (int dy = 0; dy < tilesz; dy++) {

                for (int dx = 0; dx < tilesz; dx++) {

                    uint8_t v = im->buf[(ty*tilesz+dy)*s + tx*tilesz + dx];
                    if (v < min)
                        min = v;
                    if (v > max)
                        max = v;
                }
            }

            im_max[ty*tw+tx] = max;
            im_min[ty*tw+tx] = min;
        }
    }

    // second, apply 3x3 max/min convolution to "blur" these values
    // over larger areas. This reduces artifacts due to abrupt changes
    // in the threshold value.
    if (1) {
        uint8_t *im_max_tmp = calloc(tw*th, sizeof(uint8_t));
        uint8_t *im_min_tmp = calloc(tw*th, sizeof(uint8_t));

        for (int ty = 0; ty < th; ty++) {
            for (int tx = 0; tx < tw; tx++) {
                uint8_t max = 0, min = 255;

                for (int dy = -1; dy <= 1; dy++) {
                    if (ty+dy < 0 || ty+dy >= th)
                        continue;
                    for (int dx = -1; dx <= 1; dx++) {
                        if (tx+dx < 0 || tx+dx >= tw)
                            continue;

                        uint8_t m = im_max[(ty+dy)*tw+tx+dx];
                        if (m > max)
                            max = m;
                        m = im_min[(ty+dy)*tw+tx+dx];
                        if (m < min)
                            min = m;
                    }
                }

                im_max_tmp[ty*tw + tx] = max;
                im_min_tmp[ty*tw + tx] = min;
            }
        }
        free(im_max);
        free(im_min);
        im_max = im_max_tmp;
        im_min = im_min_tmp;
    }

    for (int ty = 0; ty < th; ty++) {
        for (int tx = 0; tx < tw; tx++) {

            int min = im_min[ty*tw + tx];
            int max = im_max[ty*tw + tx];

            // low contrast region? (no edges)
            if (max - min < td->qtp.min_white_black_diff) {
                for (int dy = 0; dy < tilesz; dy++) {
                    int y = ty*tilesz + dy;

                    for (int dx = 0; dx < tilesz; dx++) {
                        int x = tx*tilesz + dx;

                        threshim->buf[y*s+x] = 127;
                    }
                }
                continue;
            }

            // otherwise, actually threshold this tile.

            // argument for biasing towards dark; specular highlights
            // can be substantially brighter than white tag parts
            uint8_t thresh = min + (max - min) / 2;

            for (int dy = 0; dy < tilesz; dy++) {
                int y = ty*tilesz + dy;

                for (int dx = 0; dx < tilesz; dx++) {
                    int x = tx*tilesz + dx;

                    uint8_t v = im->buf[y*s+x];
                    if (v > thresh)
                        threshim->buf[y*s+x] = 255;
                    else
                        threshim->buf[y*s+x] = 0;
                }
            }
        }
    }

    // we skipped over the non-full-sized tiles above. Fix those now.
    if (1) {
        for (int y = 0; y < h; y++) {

            // what is the first x coordinate we need to process in this row?

            int x0;

            if (y >= th*tilesz) {
                x0 = 0; // we're at the bottom; do the whole row.
            } else {
                x0 = tw*tilesz; // we only need to do the right most part.
            }

            // compute tile coordinates and clamp.
            int ty = y / tilesz;
            if (ty >= th)
                ty = th - 1;

            for (int x = x0; x < w; x++) {
                int tx = x / tilesz;
                if (tx >= tw)
                    tx = tw - 1;

                int max = im_max[ty*tw + tx];
                int min = im_min[ty*tw + tx];
                int thresh = min + (max - min) / 2;

                uint8_t v = im->buf[y*s+x];
                if (v > thresh)
                    threshim->buf[y*s+x] = 255;
                else
                    threshim->buf[y*s+x] = 0;
            }
        }
    }

    free(im_min);
    free(im_max);

    // this is a dilate/erode deglitching scheme that does not improve
    // anything as far as I can tell.
    if (0 || td->qtp.deglitch) {
        image_u8_t *tmp = image_u8_create(w, h);

        for (int y = 1; y + 1 < h; y++) {
            for (int x = 1; x + 1 < w; x++) {
                uint8_t max = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        uint8_t v = threshim->buf[(y+dy)*s + x + dx];
                        if (v > max)
                            max = v;
                    }
                }
                tmp->buf[y*s+x] = max;
            }
        }

        for (int y = 1; y + 1 < h; y++) {
            for (int x = 1; x + 1 < w; x++) {
                uint8_t min = 255;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        uint8_t v = tmp->buf[(y+dy)*s + x + dx];
                        if (v < min)
                            min = v;
                    }
                }
                threshim->buf[y*s+x] = min;
            }
        }

        image_u8_destroy(tmp);
    }

    timeprofile_stamp(td->tp, "threshold");

    return threshim;
}

// basically the same as threshold(), but assumes the input image is a
// bayer image. It collects statistics separately for each 2x2 block
// of pixels. NOT WELL TESTED.
image_u8_t *threshold_bayer(apriltag_detector_t *td, image_u8_t *im)
{
    int w = im->width, h = im->height, s = im->stride;

    image_u8_t *threshim = image_u8_create_alignment(w, h, s);
    assert(threshim->stride == s);

    int tilesz = 32;
    assert((tilesz & 1) == 0); // must be multiple of 2

    int tw = w/tilesz + 1;
    int th = h/tilesz + 1;

    uint8_t *im_max[4], *im_min[4];
    for (int i = 0; i < 4; i++) {
        im_max[i] = calloc(tw*th, sizeof(uint8_t));
        im_min[i] = calloc(tw*th, sizeof(uint8_t));
    }

    for (int ty = 0; ty < th; ty++) {
        for (int tx = 0; tx < tw; tx++) {

            uint8_t max[4] = { 0, 0, 0, 0};
            uint8_t min[4] = { 255, 255, 255, 255 };

            for (int dy = 0; dy < tilesz; dy++) {
                if (ty*tilesz+dy >= h)
                    continue;

                for (int dx = 0; dx < tilesz; dx++) {
                    if (tx*tilesz+dx >= w)
                        continue;

                    // which bayer element is this pixel?
                    int idx = (2*(dy&1) + (dx&1));

                    uint8_t v = im->buf[(ty*tilesz+dy)*s + tx*tilesz + dx];
                    if (v < min[idx])
                        min[idx] = v;
                    if (v > max[idx])
                        max[idx] = v;
                }
            }

            for (int i = 0; i < 4; i++) {
                im_max[i][ty*tw+tx] = max[i];
                im_min[i][ty*tw+tx] = min[i];
            }
        }
    }

    for (int ty = 0; ty < th; ty++) {
        for (int tx = 0; tx < tw; tx++) {

            uint8_t max[4] = { 0, 0, 0, 0};
            uint8_t min[4] = { 255, 255, 255, 255 };

            for (int dy = -1; dy <= 1; dy++) {
                if (ty+dy < 0 || ty+dy >= th)
                    continue;
                for (int dx = -1; dx <= 1; dx++) {
                    if (tx+dx < 0 || tx+dx >= tw)
                        continue;

                    for (int i = 0; i < 4; i++) {
                        uint8_t m = im_max[i][(ty+dy)*tw+tx+dx];
                        if (m > max[i])
                            max[i] = m;
                        m = im_min[i][(ty+dy)*tw+tx+dx];
                        if (m < min[i])
                            min[i] = m;
                    }
                }
            }

            // XXX CONSTANT
//            if (max - min < 30)
//                continue;

            // argument for biasing towards dark: specular highlights
            // can be substantially brighter than white tag parts
            uint8_t thresh[4];
            for (int i = 0; i < 4; i++) {
                thresh[i] = min[i] + (max[i] - min[i]) / 2;
            }

            for (int dy = 0; dy < tilesz; dy++) {
                int y = ty*tilesz + dy;
                if (y >= h)
                    continue;

                for (int dx = 0; dx < tilesz; dx++) {
                    int x = tx*tilesz + dx;
                    if (x >= w)
                        continue;

                    // which bayer element is this pixel?
                    int idx = (2*(y&1) + (x&1));

                    uint8_t v = im->buf[y*s+x];
                    threshim->buf[y*s+x] = v > thresh[idx];
                }
            }
        }
    }

    for (int i = 0; i < 4; i++) {
        free(im_min[i]);
        free(im_max[i]);
    }

    timeprofile_stamp(td->tp, "threshold");

    return threshim;
}

unionfind_t* connected_components(apriltag_detector_t *td, image_u8_t* threshim, int w, int h, int ts) {
    unionfind_t *uf = unionfind_create(w * h);

    if (td->nthreads <= 1) {
        do_unionfind_first_line(uf, threshim, h, w, ts);
        for (int y = 1; y < h; y++) {
            do_unionfind_line2(uf, threshim, h, w, ts, y);
        }
    } else {
        do_unionfind_first_line(uf, threshim, h, w, ts);

        int sz = h;
        int chunksize = 1 + sz / (APRILTAG_TASKS_PER_THREAD_TARGET * td->nthreads);
        struct unionfind_task *tasks = malloc(sizeof(struct unionfind_task)*(sz / chunksize + 1));

        int ntasks = 0;

        for (int i = 1; i < sz; i += chunksize) {
            // each task will process [y0, y1). Note that this attaches
            // each cell to the right and down, so row y1 *is* potentially modified.
            //
            // for parallelization, make sure that each task doesn't touch rows
            // used by another thread.
            tasks[ntasks].y0 = i;
            tasks[ntasks].y1 = imin(sz, i + chunksize - 1);
            tasks[ntasks].h = h;
            tasks[ntasks].w = w;
            tasks[ntasks].s = ts;
            tasks[ntasks].uf = uf;
            tasks[ntasks].im = threshim;

            workerpool_add_task(td->wp, do_unionfind_task2, &tasks[ntasks]);
            ntasks++;
        }

        workerpool_run(td->wp);

        // XXX stitch together the different chunks.
        for (int i = 1; i < ntasks; i++) {
            do_unionfind_line2(uf, threshim, h, w, ts, tasks[i].y0 - 1);
        }

        free(tasks);
    }
    return uf;
}

zarray_t* do_gradient_clusters(image_u8_t* threshim, int ts, int y0, int y1, int w, int nclustermap, unionfind_t* uf, zarray_t* clusters) {
    struct uint64_zarray_entry **clustermap = calloc(nclustermap, sizeof(struct uint64_zarray_entry*));

    int mem_chunk_size = 2048;
    struct uint64_zarray_entry** mem_pools = malloc(sizeof(struct uint64_zarray_entry *)*2*nclustermap/mem_chunk_size);
    int mem_pool_idx = 0;
    int mem_pool_loc = 0;
    mem_pools[mem_pool_idx] = calloc(mem_chunk_size, sizeof(struct uint64_zarray_entry));

    for (int y = y0; y < y1; y++) {
        for (int x = 1; x < w-1; x++) {

            uint8_t v0 = threshim->buf[y*ts + x];
            if (v0 == 127)
                continue;

            // XXX don't query this until we know we need it?
            uint64_t rep0 = unionfind_get_representative(uf, y*w + x);
            if (unionfind_get_set_size(uf, rep0) < 25) {
                continue;
            }

            // whenever we find two adjacent pixels such that one is
            // white and the other black, we add the point half-way
            // between them to a cluster associated with the unique
            // ids of the white and black regions.
            //
            // We additionally compute the gradient direction (i.e., which
            // direction was the white pixel?) Note: if (v1-v0) == 255, then
            // (dx,dy) points towards the white pixel. if (v1-v0) == -255, then
            // (dx,dy) points towards the black pixel. p.gx and p.gy will thus
            // be -255, 0, or 255.
            //
            // Note that any given pixel might be added to multiple
            // different clusters. But in the common case, a given
            // pixel will be added multiple times to the same cluster,
            // which increases the size of the cluster and thus the
            // computational costs.
            //
            // A possible optimization would be to combine entries
            // within the same cluster.

#define DO_CONN(dx, dy)                                                 \
            if (1) {                                                    \
                uint8_t v1 = threshim->buf[(y + dy)*ts + x + dx];       \
                                                                        \
                if (v0 + v1 == 255) {                                   \
                    uint64_t rep1 = unionfind_get_representative(uf, (y + dy)*w + x + dx); \
                    if (unionfind_get_set_size(uf, rep1) > 24) {        \
                        uint64_t clusterid;                                 \
                        if (rep0 < rep1)                                    \
                            clusterid = (rep1 << 32) + rep0;                \
                        else                                                \
                            clusterid = (rep0 << 32) + rep1;                \
                                                                            \
                        /* XXX lousy hash function */                       \
                        uint32_t clustermap_bucket = u64hash_2(clusterid) % nclustermap; \
                        struct uint64_zarray_entry *entry = clustermap[clustermap_bucket]; \
                        while (entry && entry->id != clusterid) {           \
                            entry = entry->next;                            \
                        }                                                   \
                                                                            \
                        if (!entry) {                                       \
                            if (mem_pool_loc == mem_chunk_size) {           \
                                mem_pool_loc = 0;                           \
                                mem_pool_idx++;                             \
                                mem_pools[mem_pool_idx] = calloc(mem_chunk_size, sizeof(struct uint64_zarray_entry)); \
                            }                                               \
                            entry = mem_pools[mem_pool_idx] + mem_pool_loc; \
                            mem_pool_loc++;                                 \
                                                                            \
                            entry->id = clusterid;                          \
                            entry->cluster = zarray_create(sizeof(struct pt)); \
                            entry->next = clustermap[clustermap_bucket];    \
                            clustermap[clustermap_bucket] = entry;          \
                        }                                                   \
                                                                            \
                        struct pt p = { .x = 2*x + dx, .y = 2*y + dy, .gx = dx*((int) v1-v0), .gy = dy*((int) v1-v0)}; \
                        zarray_add(entry->cluster, &p);                     \
                    }                                                   \
                }                                                       \
            }

            // do 4 connectivity. NB: Arguments must be [-1, 1] or we'll overflow .gx, .gy
            DO_CONN(1, 0);
            DO_CONN(0, 1);

            // do 8 connectivity
            DO_CONN(-1, 1);
            DO_CONN(1, 1);
        }
    }
#undef DO_CONN

    for (int i = 0; i < nclustermap; i++) {
        int start = zarray_size(clusters);
        for (struct uint64_zarray_entry *entry = clustermap[i]; entry; entry = entry->next) {
            struct cluster_hash* cluster_hash = malloc(sizeof(struct cluster_hash));
            cluster_hash->hash = u64hash_2(entry->id) % nclustermap;
            cluster_hash->id = entry->id;
            cluster_hash->data = entry->cluster;
            zarray_add(clusters, &cluster_hash);
        }
        int end = zarray_size(clusters);

        // Do a quick bubblesort on the secondary key.
        int n = end - start;
        for (int j = 0; j < n - 1; j++) {
            for (int k = 0; k < n - j - 1; k++) {
                struct cluster_hash* hash1;
                struct cluster_hash* hash2;
                zarray_get(clusters, start + k, &hash1);
                zarray_get(clusters, start + k + 1, &hash2);
                if (hash1->id > hash2->id) {
                    struct cluster_hash tmp = *hash2;
                    *hash2 = *hash1;
                    *hash1 = tmp;
                }
            }
        }
    }
    for (int i = 0; i <= mem_pool_idx; i++) {
        free(mem_pools[i]);
    }
    free(mem_pools);
    free(clustermap);

    return clusters;
}

static void do_cluster_task(void *p)
{
    struct cluster_task *task = (struct cluster_task*) p;

    do_gradient_clusters(task->im, task->s, task->y0, task->y1, task->w, task->nclustermap, task->uf, task->clusters);
}

zarray_t* merge_clusters(zarray_t* c1, zarray_t* c2) {
    zarray_t* ret = zarray_create(sizeof(struct cluster_hash*));
    zarray_ensure_capacity(ret, zarray_size(c1) + zarray_size(c2));

    int i1 = 0;
    int i2 = 0;
    int l1 = zarray_size(c1);
    int l2 = zarray_size(c2);

    while (i1 < l1 && i2 < l2) {
        struct cluster_hash* h1;
        struct cluster_hash* h2;
        zarray_get(c1, i1, &h1);
        zarray_get(c2, i2, &h2);

        if (h1->hash == h2->hash && h1->id == h2->id) {
            zarray_add_all(h1->data, h2->data);
            zarray_add(ret, &h1);
            i1++;
            i2++;
            zarray_destroy(h2->data);
            free(h2);
        } else if (h2->hash < h1->hash || (h2->hash == h1->hash && h2->id < h1->id)) {
            zarray_add(ret, &h2);
            i2++;
        } else {
            zarray_add(ret, &h1);
            i1++;
        }
    }

    for (; i1 < l1; i1++) {
        struct cluster_hash* h1;
        zarray_get(c1, i1, &h1);
        zarray_add(ret, &h1);
    }

    for (; i2 < l2; i2++) {
        struct cluster_hash* h2;
        zarray_get(c2, i2, &h2);
        zarray_add(ret, &h2);
    }

    zarray_destroy(c1);
    zarray_destroy(c2);

    return ret;
}

zarray_t* gradient_clusters(apriltag_detector_t *td, image_u8_t* threshim, int w, int h, int ts, unionfind_t* uf) {
    zarray_t* clusters;
    int nclustermap = 0.2*w*h;

    int sz = h - 1;
    int chunksize = 1 + sz / td->nthreads;
    struct cluster_task *tasks = malloc(sizeof(struct cluster_task)*(sz / chunksize + 1));

    int ntasks = 0;

    for (int i = 1; i < sz; i += chunksize) {
        // each task will process [y0, y1). Note that this processes
        // each cell to the right and down.
        tasks[ntasks].y0 = i;
        tasks[ntasks].y1 = imin(sz, i + chunksize);
        tasks[ntasks].w = w;
        tasks[ntasks].s = ts;
        tasks[ntasks].uf = uf;
        tasks[ntasks].im = threshim;
        tasks[ntasks].nclustermap = nclustermap/(sz / chunksize + 1);
        tasks[ntasks].clusters = zarray_create(sizeof(struct cluster_hash*));

        workerpool_add_task(td->wp, do_cluster_task, &tasks[ntasks]);
        ntasks++;
    }

    workerpool_run(td->wp);

    zarray_t** clusters_list = malloc(sizeof(zarray_t *)*ntasks);
    for (int i = 0; i < ntasks; i++) {
        clusters_list[i] = tasks[i].clusters;
    }

    int length = ntasks;
    while (length > 1) {
        int write = 0;
        for (int i = 0; i < length - 1; i += 2) {
            clusters_list[write] = merge_clusters(clusters_list[i], clusters_list[i + 1]);
            write++;
        }

        if (length % 2) {
            clusters_list[write] = clusters_list[length - 1];
        }

        length = (length >> 1) + length % 2;
    }

    clusters = zarray_create(sizeof(zarray_t*));
    zarray_ensure_capacity(clusters, zarray_size(clusters_list[0]));
    for (int i = 0; i < zarray_size(clusters_list[0]); i++) {
        struct cluster_hash* h;
        zarray_get(clusters_list[0], i, &h);
        zarray_add(clusters, &h->data);
        free(h);
    }
    zarray_destroy(clusters_list[0]);
    free(clusters_list);
    free(tasks);
    return clusters;
}

zarray_t* fit_quads(apriltag_detector_t *td, int w, int h, zarray_t* clusters, image_u8_t* im) {
    zarray_t *quads = zarray_create(sizeof(struct quad));

    bool normal_border = false;
    bool reversed_border = false;
    int min_tag_width = 1000000;
    for (int i = 0; i < zarray_size(td->tag_families); i++) {
        apriltag_family_t* family;
        zarray_get(td->tag_families, i, &family);
        if (family->width_at_border < min_tag_width) {
            min_tag_width = family->width_at_border;
        }
        normal_border |= !family->reversed_border;
        reversed_border |= family->reversed_border;
    }
    min_tag_width /= td->quad_decimate;
    if (min_tag_width < 3) {
        min_tag_width = 3;
    }

    int sz = zarray_size(clusters);
    int chunksize = 1 + sz / (APRILTAG_TASKS_PER_THREAD_TARGET * td->nthreads);
    struct quad_task *tasks = malloc(sizeof(struct quad_task)*(sz / chunksize + 1));

    int ntasks = 0;
    for (int i = 0; i < sz; i += chunksize) {
        tasks[ntasks].td = td;
        tasks[ntasks].cidx0 = i;
        tasks[ntasks].cidx1 = imin(sz, i + chunksize);
        tasks[ntasks].h = h;
        tasks[ntasks].w = w;
        tasks[ntasks].quads = quads;
        tasks[ntasks].clusters = clusters;
        tasks[ntasks].im = im;
        tasks[ntasks].tag_width = min_tag_width;
        tasks[ntasks].normal_border = normal_border;
        tasks[ntasks].reversed_border = reversed_border;

        workerpool_add_task(td->wp, do_quad_task, &tasks[ntasks]);
        ntasks++;
    }

    workerpool_run(td->wp);

    free(tasks);

    return quads;
}

zarray_t *apriltag_quad_thresh(apriltag_detector_t *td, image_u8_t *im)
{
    ////////////////////////////////////////////////////////
    // step 1. threshold the image, creating the edge image.

    int w = im->width, h = im->height;

    image_u8_t *threshim = threshold(td, im);
    int ts = threshim->stride;

    if (td->debug)
        image_u8_write_pnm(threshim, "debug_threshold.pnm");


    ////////////////////////////////////////////////////////
    // step 2. find connected components.
    unionfind_t* uf = connected_components(td, threshim, w, h, ts);

    // make segmentation image.
    if (td->debug) {
        image_u8x3_t *d = image_u8x3_create(w, h);

        uint32_t *colors = (uint32_t*) calloc(w*h, sizeof(*colors));

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                uint32_t v = unionfind_get_representative(uf, y*w+x);

                if (unionfind_get_set_size(uf, v) < td->qtp.min_cluster_pixels)
                    continue;

                uint32_t color = colors[v];
                uint8_t r = color >> 16,
                    g = color >> 8,
                    b = color;

                if (color == 0) {
                    const int bias = 50;
                    r = bias + (random() % (200-bias));
                    g = bias + (random() % (200-bias));
                    b = bias + (random() % (200-bias));
                    colors[v] = (r << 16) | (g << 8) | b;
                }

                d->buf[y*d->stride + 3*x + 0] = r;
                d->buf[y*d->stride + 3*x + 1] = g;
                d->buf[y*d->stride + 3*x + 2] = b;
            }
        }

        free(colors);

        image_u8x3_write_pnm(d, "debug_segmentation.pnm");
        image_u8x3_destroy(d);
    }


    timeprofile_stamp(td->tp, "unionfind");

    zarray_t* clusters = gradient_clusters(td, threshim, w, h, ts, uf);

    if (td->debug) {
        image_u8x3_t *d = image_u8x3_create(w, h);

        for (int i = 0; i < zarray_size(clusters); i++) {
            zarray_t *cluster;
            zarray_get(clusters, i, &cluster);

            uint32_t r, g, b;

            if (1) {
                const int bias = 50;
                r = bias + (random() % (200-bias));
                g = bias + (random() % (200-bias));
                b = bias + (random() % (200-bias));
            }

            for (int j = 0; j < zarray_size(cluster); j++) {
                struct pt *p;
                zarray_get_volatile(cluster, j, &p);

                int x = p->x / 2;
                int y = p->y / 2;
                d->buf[y*d->stride + 3*x + 0] = r;
                d->buf[y*d->stride + 3*x + 1] = g;
                d->buf[y*d->stride + 3*x + 2] = b;
            }
        }

        image_u8x3_write_pnm(d, "debug_clusters.pnm");
        image_u8x3_destroy(d);
    }


    image_u8_destroy(threshim);
    timeprofile_stamp(td->tp, "make clusters");

    ////////////////////////////////////////////////////////
    // step 3. process each connected component.

    zarray_t* quads = fit_quads(td, w, h, clusters, im);

    if (td->debug) {
        FILE *f = fopen("debug_lines.ps", "w");
        fprintf(f, "%%!PS\n\n");

        image_u8_t *im2 = image_u8_copy(im);
        image_u8_darken(im2);
        image_u8_darken(im2);

        // assume letter, which is 612x792 points.
        double scale = fmin(612.0/im->width, 792.0/im2->height);
        fprintf(f, "%.15f %.15f scale\n", scale, scale);
        fprintf(f, "0 %d translate\n", im2->height);
        fprintf(f, "1 -1 scale\n");

        postscript_image(f, im2);

        image_u8_destroy(im2);

        for (int i = 0; i < zarray_size(quads); i++) {
            struct quad *q;
            zarray_get_volatile(quads, i, &q);

            float rgb[3];
            int bias = 100;

            for (int i = 0; i < 3; i++)
                rgb[i] = bias + (random() % (255-bias));

            fprintf(f, "%f %f %f setrgbcolor\n", rgb[0]/255.0f, rgb[1]/255.0f, rgb[2]/255.0f);
            fprintf(f, "%.15f %.15f moveto %.15f %.15f lineto %.15f %.15f lineto %.15f %.15f lineto %.15f %.15f lineto stroke\n",
                    q->p[0][0], q->p[0][1],
                    q->p[1][0], q->p[1][1],
                    q->p[2][0], q->p[2][1],
                    q->p[3][0], q->p[3][1],
                    q->p[0][0], q->p[0][1]);
        }

        fclose(f);
    }

    timeprofile_stamp(td->tp, "fit quads to clusters");

    unionfind_destroy(uf);

    for (int i = 0; i < zarray_size(clusters); i++) {
        zarray_t *cluster;
        zarray_get(clusters, i, &cluster);
        zarray_destroy(cluster);
    }
    zarray_destroy(clusters);

    return quads;
}
