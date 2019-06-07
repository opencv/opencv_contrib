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

#pragma once

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <string.h> // memcpy

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_TWOPI
# define M_TWOPI       6.2831853071795862319959  /* 2*pi */
#endif

#ifndef M_PI
# define M_PI 3.141592653589793238462643383279502884196
#endif

#define to_radians(x) ( (x) * (M_PI / 180.0 ))
#define to_degrees(x) ( (x) * (180.0 / M_PI ))

#define max(A, B) (A < B ? B : A)
#define min(A, B) (A < B ? A : B)

  /* DEPRECATE, threshold meaningless without context.
static inline int dequals(double a, double b)
{
    double thresh = 1e-9;
    return (fabs(a-b) < thresh);
}
  */

static inline int dequals_mag(double a, double b, double thresh)
{
    return (fabs(a-b) < thresh);
}

static inline int isq(int v)
{
    return v*v;
}

static inline float fsq(float v)
{
    return v*v;
}

static inline double sq(double v)
{
    return v*v;
}

static inline double sgn(double v)
{
    return (v>=0) ? 1 : -1;
}

// random number between [0, 1)
static inline float randf()
{
    return (float)(rand() / (RAND_MAX + 1.0));
}


static inline float signed_randf()
{
    return randf()*2 - 1;
}

// return a random integer between [0, bound)
static inline int irand(int bound)
{
    int v = (int) (randf()*bound);
    if (v == bound)
        return (bound-1);
    //assert(v >= 0);
    //assert(v < bound);
    return v;
}

/** Map vin to [0, 2*PI) **/
static inline double mod2pi_positive(double vin)
{
    return vin - M_TWOPI * floor(vin / M_TWOPI);
}

/** Map vin to [-PI, PI) **/
static inline double mod2pi(double vin)
{
    return mod2pi_positive(vin + M_PI) - M_PI;
}

/** Return vin such that it is within PI degrees of ref **/
static inline double mod2pi_ref(double ref, double vin)
{
    return ref + mod2pi(vin - ref);
}

/** Map vin to [0, 360) **/
static inline double mod360_positive(double vin)
{
    return vin - 360 * floor(vin / 360);
}

/** Map vin to [-180, 180) **/
static inline double mod360(double vin)
{
    return mod360_positive(vin + 180) - 180;
}

static inline int mod_positive(int vin, int mod) {
    return (vin % mod + mod) % mod;
}

static inline int theta_to_int(double theta, int max)
{
    theta = mod2pi_ref(M_PI, theta);
    int v = (int) (theta / M_TWOPI * max);

    if (v == max)
        v = 0;

    assert (v >= 0 && v < max);

    return v;
}

static inline int imin(int a, int b)
{
    return (a < b) ? a : b;
}

static inline int imax(int a, int b)
{
    return (a > b) ? a : b;
}

static inline int64_t imin64(int64_t a, int64_t b)
{
    return (a < b) ? a : b;
}

static inline int64_t imax64(int64_t a, int64_t b)
{
    return (a > b) ? a : b;
}

static inline int iclamp(int v, int minv, int maxv)
{
    return imax(minv, imin(v, maxv));
}

static inline double dclamp(double a, double min, double max)
{
    if (a < min)
        return min;
    if (a > max)
        return max;
    return a;
}

static inline int fltcmp (float f1, float f2)
{
    float epsilon = f1-f2;
    if (epsilon < 0.0)
        return -1;
    else if (epsilon > 0.0)
        return  1;
    else
        return  0;
}

static inline int dblcmp (double d1, double d2)
{
    double epsilon = d1-d2;
    if (epsilon < 0.0)
        return -1;
    else if (epsilon > 0.0)
        return  1;
    else
        return  0;
}

#ifdef __cplusplus
}
#endif
