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

#ifndef _OPENCV_APRIL_QUAD_THRESH_HPP_
#define _OPENCV_APRIL_QUAD_THRESH_HPP_
#ifdef __cplusplus

#include "precomp.hpp"
#include "opencv2/aruco.hpp"
#include "unionfind.hpp"
#include "zmaxheap.hpp"
#include "zarray.hpp"

namespace cv {
namespace aruco {

static inline uint32_t u64hash_2(uint64_t x) {
    return (2654435761 * x) >> 32;
    return (uint32_t) x;
}

struct uint64_zarray_entry{
    uint64_t id;
    zarray_t *cluster;

    struct uint64_zarray_entry *next;
};

struct pt{
    // Note: these represent 2*actual value.
    uint16_t x, y;
    float theta;
    int16_t gx, gy;
};

struct remove_vertex{
    int i;           // which vertex to remove?
    int left, right; // left vertex, right vertex

    double err;
};

struct segment{
    int is_vertex;

    // always greater than zero, but right can be > size, which denotes
    // a wrap around back to the beginning of the points. and left < right.
    int left, right;
};

struct line_fit_pt{
    double Mx, My;
    double Mxx, Myy, Mxy;
    double W; // total weight
};

static inline void ptsort(struct pt *pts, int sz);

/**
 * lfps contains *cumulative* moments for N points, with
 * index j reflecting points [0,j] (inclusive).
 * fit a line to the points [i0, i1] (inclusive). i0, i1 are both (0, sz)
 * if i1 < i0, we treat this as a wrap around.
 */
void fit_line(struct line_fit_pt *lfps, int sz, int i0, int i1, double *lineparm, double *err, double *mse);

int err_compare_descending(const void *_a, const void *_b);

/**
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
 **/
int quad_segment_maxima(const Ptr<DetectorParameters> &td, int sz, struct line_fit_pt *lfps, int indices[4]);

/**
 * returns 0 if the cluster looks bad.
 */
int quad_segment_agg(int sz, struct line_fit_pt *lfps, int indices[4]);

/**
 *  return 1 if the quad looks okay, 0 if it should be discarded
 *  quad
 **/
int fit_quad(const Ptr<DetectorParameters> &_params, const Mat im, zarray_t *cluster, struct sQuad *quad);

/**
 *
 * @param nCidx0
 * @param nCidx1
 * @param nClusters
 * @param nW
 * @param nH
 * @param nquads
 * @param td
 * @param im
 */
static void do_quad(int nCidx0, int nCidx1, zarray_t &nClusters, int nW, int nH, zarray_t *nquads, const Ptr<DetectorParameters> &td, const Mat im);

/**
 *
 * @param mIm
 * @param parameters
 * @param mThresh
 */
void threshold(const Mat mIm, const Ptr<DetectorParameters> &parameters, Mat& mThresh);

/**
 *
 * @param parameters
 * @param mImg
 * @param contours
 * @return
 */
zarray_t *apriltag_quad_thresh(const Ptr<DetectorParameters> &parameters, const Mat & mImg, std::vector< std::vector< Point > > &contours);

}}
#endif
#endif
