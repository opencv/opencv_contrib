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

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>

#include "matd.h"
#include "math_util.h"

// XXX Write unit tests for me!
// XXX Rewrite matd_coords in terms of this.

/*
  This file provides conversions between the following formats:

  quaternion (TNAME[4], { w, x, y, z})

  xyt  (translation in x, y, and rotation in radians.)

  xytcov  (xyt as a TNAME[3] followed by covariance TNAME[9])

  xy, xyz  (translation in x, y, and z)

  mat44 (4x4 rigid-body transformation matrix, row-major
  order. Conventions: We assume points are projected via right
  multiplication. E.g., p' = Mp.) Note: some functions really do rely
  on it being a RIGID, scale=1 transform.

  angleaxis (TNAME[4], { angle-rads, x, y, z }

  xyzrpy (translation x, y, z, euler angles)

  Roll Pitch Yaw are evaluated in the order: roll, pitch, then yaw. I.e.,
  rollPitchYawToMatrix(rpy) = rotateZ(rpy[2]) * rotateY(rpy[1]) * Rotatex(rpy[0])
*/

#define TRRFN(root, suffix) root ## suffix
#define TRFN(root, suffix) TRRFN(root, suffix)
#define TFN(suffix) TRFN(TNAME, suffix)

// if V is null, returns null.
static inline TNAME *TFN(s_dup)(const TNAME *v, int len)
{
    if (!v)
        return NULL;

    TNAME *r = (TNAME*)malloc(len * sizeof(TNAME));
    memcpy(r, v, len * sizeof(TNAME));
    return r;
}

static inline void TFN(s_print)(const TNAME *a, int len, const char *fmt)
{
    for (int i = 0; i < len; i++)
        printf(fmt, a[i]);
    printf("\n");
}

static inline void TFN(s_print_mat)(const TNAME *a, int nrows, int ncols, const char *fmt)
{
    for (int i = 0; i < nrows * ncols; i++) {
        printf(fmt, a[i]);
        if ((i % ncols) == (ncols - 1))
            printf("\n");
    }
}

static inline void TFN(s_print_mat44)(const TNAME *a, const char *fmt)
{
    for (int i = 0; i < 4 * 4; i++) {
        printf(fmt, a[i]);
        if ((i % 4) == 3)
            printf("\n");
    }
}

static inline void TFN(s_add)(const TNAME *a, const TNAME *b, int len, TNAME *r)
{
    for (int i = 0; i < len; i++)
        r[i] = a[i] + b[i];
}

static inline void TFN(s_subtract)(const TNAME *a, const TNAME *b, int len, TNAME *r)
{
    for (int i = 0; i < len; i++)
        r[i] = a[i] - b[i];
}

static inline void TFN(s_scale)(TNAME s, const TNAME *v, int len, TNAME *r)
{
    for (int i = 0; i < len; i++)
        r[i] = s * v[i];
}

static inline TNAME TFN(s_dot)(const TNAME *a, const TNAME *b, int len)
{
    TNAME acc = 0;
    for (int i = 0; i < len; i++)
        acc += a[i] * b[i];
    return acc;
}

static inline TNAME TFN(s_distance)(const TNAME *a, const TNAME *b, int len)
{
    TNAME acc = 0;
    for (int i = 0; i < len; i++)
        acc += (a[i] - b[i])*(a[i] - b[i]);
    return (TNAME)sqrt(acc);
}

static inline TNAME TFN(s_squared_distance)(const TNAME *a, const TNAME *b, int len)
{
    TNAME acc = 0;
    for (int i = 0; i < len; i++)
        acc += (a[i] - b[i])*(a[i] - b[i]);
    return acc;
}

static inline TNAME TFN(s_squared_magnitude)(const TNAME *v, int len)
{
    TNAME acc = 0;
    for (int i = 0; i < len; i++)
        acc += v[i]*v[i];
    return acc;
}

static inline TNAME TFN(s_magnitude)(const TNAME *v, int len)
{
    TNAME acc = 0;
    for (int i = 0; i < len; i++)
        acc += v[i]*v[i];
    return (TNAME)sqrt(acc);
}

static inline void TFN(s_normalize)(const TNAME *v, int len, TNAME *r)
{
    TNAME mag = TFN(s_magnitude)(v, len);
    for (int i = 0; i < len; i++)
        r[i] = v[i] / mag;
}

static inline void TFN(s_normalize_self)(TNAME *v, int len)
{
    TNAME mag = TFN(s_magnitude)(v, len);
    for (int i = 0; i < len; i++)
        v[i] /= mag;
}

static inline void TFN(s_scale_self)(TNAME *v, int len, double scale)
{
    for (int i = 0; i < len; i++)
        v[i] = (TNAME)(v[i] * scale);
}

static inline void TFN(s_quat_rotate)(const TNAME q[4], const TNAME v[3], TNAME r[3])
{
    TNAME t2, t3, t4, t5, t6, t7, t8, t9, t10;

    t2 = q[0]*q[1];
    t3 = q[0]*q[2];
    t4 = q[0]*q[3];
    t5 = -q[1]*q[1];
    t6 = q[1]*q[2];
    t7 = q[1]*q[3];
    t8 = -q[2]*q[2];
    t9 = q[2]*q[3];
    t10 = -q[3]*q[3];

    r[0] = 2*((t8+t10)*v[0] + (t6-t4)*v[1]  + (t3+t7)*v[2]) + v[0];
    r[1] = 2*((t4+t6)*v[0]  + (t5+t10)*v[1] + (t9-t2)*v[2]) + v[1];
    r[2] = 2*((t7-t3)*v[0]  + (t2+t9)*v[1]  + (t5+t8)*v[2]) + v[2];
}

static inline void TFN(s_quat_multiply)(const TNAME a[4], const TNAME b[4], TNAME r[4])
{
    r[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    r[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    r[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    r[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}

static inline void TFN(s_quat_inverse)(const TNAME q[4], TNAME r[4])
{
    TNAME mag = TFN(s_magnitude)(q, 4);
    r[0] = q[0]/mag;
    r[1] = -q[1]/mag;
    r[2] = -q[2]/mag;
    r[3] = -q[3]/mag;
}

static inline void TFN(s_copy)(const TNAME *src, TNAME *dst, int n)
{
    memcpy(dst, src, n * sizeof(TNAME));
}

static inline void TFN(s_xyt_copy)(const TNAME xyt[3], TNAME r[3])
{
    TFN(s_copy)(xyt, r, 3);
}

static inline void TFN(s_xyt_to_mat44)(const TNAME xyt[3], TNAME r[16])
{
    TNAME s = (TNAME)sin(xyt[2]), c = (TNAME)cos(xyt[2]);
    memset(r, 0, sizeof(TNAME)*16);
    r[0] = c;
    r[1] = -s;
    r[3] = xyt[0];
    r[4] = s;
    r[5] = c;
    r[7] = xyt[1];
    r[10] = 1;
    r[15] = 1;
}

static inline void TFN(s_xyt_transform_xy)(const TNAME xyt[3], const TNAME xy[2], TNAME r[2])
{
    TNAME s = (TNAME)sin(xyt[2]), c = (TNAME)cos(xyt[2]);
    r[0] = c*xy[0] - s*xy[1] + xyt[0];
    r[1] = s*xy[0] + c*xy[1] + xyt[1];
}

static inline void TFN(s_mat_transform_xyz)(const TNAME M[16], const TNAME xyz[3], TNAME r[3])
{
    r[0] = M[0]*xyz[0] + M[1]*xyz[1] + M[2]*xyz[2]  + M[3];
    r[1] = M[4]*xyz[0] + M[5]*xyz[1] + M[6]*xyz[2]  + M[7];
    r[2] = M[8]*xyz[0] + M[9]*xyz[1] + M[10]*xyz[2] + M[11];
}

static inline void TFN(s_quat_to_angleaxis)(const TNAME _q[4], TNAME r[4])
{
    TNAME q[4];
    TFN(s_normalize)(_q, 4, q);

    // be polite: return an angle from [-pi, pi]
    // use atan2 to be 4-quadrant safe
    TNAME mag = TFN(s_magnitude)(&q[1], 3);
    r[0] = (TNAME)mod2pi(2 * atan2(mag, q[0]));
    if (mag != 0) {
        r[1] = q[1] / mag;
        r[2] = q[2] / mag;
        r[3] = q[3] / mag;
    } else {
        r[1] = 1;
        r[2] = 0;
        r[3] = 0;
    }
}

static inline void TFN(s_angleaxis_to_quat)(const TNAME aa[4], TNAME q[4])
{
    TNAME rad = aa[0];
    q[0] = (TNAME)cos(rad / 2.0);
    TNAME s = (TNAME)sin(rad / 2.0);

    TNAME v[3] = { aa[1], aa[2], aa[3] };
    TFN(s_normalize)(v, 3, v);

    q[1] = s * v[0];
    q[2] = s * v[1];
    q[3] = s * v[2];
}

static inline void TFN(s_quat_to_mat44)(const TNAME q[4], TNAME r[16])
{
    TNAME w = q[0], x = q[1], y = q[2], z = q[3];

    r[0] = w*w + x*x - y*y - z*z;
    r[1] = 2*x*y - 2*w*z;
    r[2] = 2*x*z + 2*w*y;
    r[3] = 0;

    r[4] = 2*x*y + 2*w*z;
    r[5] = w*w - x*x + y*y - z*z;
    r[6] = 2*y*z - 2*w*x;
    r[7] = 0;

    r[8] = 2*x*z - 2*w*y;
    r[9] = 2*y*z + 2*w*x;
    r[10] = w*w - x*x - y*y + z*z;
    r[11] = 0;

    r[12] = 0;
    r[13] = 0;
    r[14] = 0;
    r[15] = 1;
}

/* Returns the skew-symmetric matrix V such that V*w = v x w (cross product).
   Sometimes denoted [v]_x or \hat{v}.
   [  0 -v3  v2
     v3   0 -v1
    -v2  v1   0]
 */
static inline void TFN(s_cross_matrix)(const TNAME v[3], TNAME V[9])
{
    V[0] = 0;
    V[1] = -v[2];
    V[2] = v[1];
    V[3] = v[2];
    V[4] = 0;
    V[5] = -v[0];
    V[6] = -v[1];
    V[7] = v[0];
    V[8] = 0;
}

static inline void TFN(s_angleaxis_to_mat44)(const TNAME aa[4], TNAME r[16])
{
    TNAME q[4];

    TFN(s_angleaxis_to_quat)(aa, q);
    TFN(s_quat_to_mat44)(q, r);
}

static inline void TFN(s_quat_xyz_to_mat44)(const TNAME q[4], const TNAME xyz[3], TNAME r[16])
{
    TFN(s_quat_to_mat44)(q, r);

    if (xyz != NULL) {
        r[3] = xyz[0];
        r[7] = xyz[1];
        r[11] = xyz[2];
    }
}

static inline void TFN(s_rpy_to_quat)(const TNAME rpy[3], TNAME quat[4])
{
    TNAME roll = rpy[0], pitch = rpy[1], yaw = rpy[2];

    TNAME halfroll = roll / 2;
    TNAME halfpitch = pitch / 2;
    TNAME halfyaw = yaw / 2;

    TNAME sin_r2 = (TNAME)sin(halfroll);
    TNAME sin_p2 = (TNAME)sin(halfpitch);
    TNAME sin_y2 = (TNAME)sin(halfyaw);

    TNAME cos_r2 = (TNAME)cos(halfroll);
    TNAME cos_p2 = (TNAME)cos(halfpitch);
    TNAME cos_y2 = (TNAME)cos(halfyaw);

    quat[0] = cos_r2 * cos_p2 * cos_y2 + sin_r2 * sin_p2 * sin_y2;
    quat[1] = sin_r2 * cos_p2 * cos_y2 - cos_r2 * sin_p2 * sin_y2;
    quat[2] = cos_r2 * sin_p2 * cos_y2 + sin_r2 * cos_p2 * sin_y2;
    quat[3] = cos_r2 * cos_p2 * sin_y2 - sin_r2 * sin_p2 * cos_y2;
}

// Reference: "A tutorial on SE(3) transformation parameterizations and
// on-manifold optimization" by Jose-Luis Blanco
static inline void TFN(s_quat_to_rpy)(const TNAME q[4], TNAME rpy[3])
{
    const TNAME qr = q[0];
    const TNAME qx = q[1];
    const TNAME qy = q[2];
    const TNAME qz = q[3];

    TNAME disc = qr*qy - qx*qz;

    if (fabs(disc+0.5) < DBL_EPSILON) {         // near -1/2
        rpy[0] = 0;
        rpy[1] = (TNAME)(-M_PI/2);
        rpy[2] = (TNAME)(2 * atan2(qx, qr));
    }
    else if (fabs(disc-0.5) < DBL_EPSILON) {    // near  1/2
        rpy[0] = 0;
        rpy[1] = (TNAME)(M_PI/2);
        rpy[2] = (TNAME)(-2 * atan2(qx, qr));
    }
    else {
        // roll
        TNAME roll_a = 2 * (qr*qx + qy*qz);
        TNAME roll_b = 1 - 2 * (qx*qx + qy*qy);
        rpy[0] = (TNAME)atan2(roll_a, roll_b);

        // pitch
        rpy[1] = (TNAME)asin(2*disc);

        // yaw
        TNAME yaw_a = 2 * (qr*qz + qx*qy);
        TNAME yaw_b = 1 - 2 * (qy*qy + qz*qz);
        rpy[2] = (TNAME)atan2(yaw_a, yaw_b);
    }
}

static inline void TFN(s_rpy_to_mat44)(const TNAME rpy[3], TNAME M[16])
{
    TNAME q[4];
    TFN(s_rpy_to_quat)(rpy, q);
    TFN(s_quat_to_mat44)(q, M);
}


static inline void TFN(s_xyzrpy_to_mat44)(const TNAME xyzrpy[6], TNAME M[16])
{
    TFN(s_rpy_to_mat44)(&xyzrpy[3], M);
    M[3] = xyzrpy[0];
    M[7] = xyzrpy[1];
    M[11] = xyzrpy[2];
}

static inline void TFN(s_mat44_transform_xyz)(const TNAME M[16], const TNAME in[3], TNAME out[3])
{
    for (int i = 0; i < 3; i++)
        out[i] = M[4*i + 0]*in[0] + M[4*i + 1]*in[1] + M[4*i + 2]*in[2] + M[4*i + 3];
}

// out = (upper 3x3 of M) * in
static inline void TFN(s_mat44_rotate_vector)(const TNAME M[16], const TNAME in[3], TNAME out[3])
{
    for (int i = 0; i < 3; i++)
        out[i] = M[4*i + 0]*in[0] + M[4*i + 1]*in[1] + M[4*i + 2]*in[2];
}

static inline void TFN(s_mat44_to_xyt)(const TNAME M[16], TNAME xyt[3])
{
    // c -s
    // s  c
    xyt[0] = M[3];
    xyt[1] = M[7];
    xyt[2] = (TNAME)atan2(M[4], M[0]);
}

static inline void TFN(s_mat_to_xyz)(const TNAME M[16], TNAME xyz[3])
{
    xyz[0] = M[3];
    xyz[1] = M[7];
    xyz[2] = M[11];
}

static inline void TFN(s_mat_to_quat)(const TNAME M[16], TNAME q[4])
{
    double T = M[0] + M[5] + M[10] + 1.0;
    double S;

    if (T > 0.0000001) {
        S = sqrt(T) * 2;
        q[0] = (TNAME)(0.25 * S);
        q[1] = (TNAME)((M[9] - M[6]) / S);
        q[2] = (TNAME)((M[2] - M[8]) / S);
        q[3] = (TNAME)((M[4] - M[1]) / S);
    } else if (M[0] > M[5] && M[0] > M[10]) {   // Column 0:
        S = sqrt(1.0 + M[0] - M[5] - M[10]) * 2;
        q[0] = (TNAME)((M[9] - M[6]) / S);
        q[1] = (TNAME)(0.25 * S);
        q[2] = (TNAME)((M[4] + M[1]) / S);
        q[3] = (TNAME)((M[2] + M[8]) / S);
    } else if (M[5] > M[10]) {                  // Column 1:
        S = sqrt(1.0 + M[5] - M[0] - M[10]) * 2;
        q[0] = (TNAME)((M[2] - M[8]) / S);
        q[1] = (TNAME)((M[4] + M[1]) / S);
        q[2] = (TNAME)(0.25 * S);
        q[3] = (TNAME)((M[9] + M[6]) / S);
    } else {                                    // Column 2:
        S = sqrt(1.0 + M[10] - M[0] - M[5]);
        q[0] = (TNAME)((M[4] - M[1]) / S);
        q[1] = (TNAME)((M[2] + M[8]) / S);
        q[2] = (TNAME)((M[9] + M[6]) / S);
        q[3] = (TNAME)(0.25 * S);
    }

    TFN(s_normalize)(q, 4, q);
}

static inline void TFN(s_quat_xyz_to_xyt)(const TNAME q[4], const TNAME xyz[3], TNAME xyt[3])
{
    TNAME M[16];
    TFN(s_quat_xyz_to_mat44)(q, xyz, M);
    TFN(s_mat44_to_xyt)(M, xyt);
}

// xytr = xyta * xytb;
static inline void TFN(s_xyt_mul)(const TNAME xyta[3], const TNAME xytb[3], TNAME xytr[3])
{
    TNAME xa = xyta[0], ya = xyta[1], ta = xyta[2];
    TNAME s = (TNAME)sin(ta), c = (TNAME)cos(ta);

    xytr[0] = c*xytb[0] - s*xytb[1] + xa;
    xytr[1] = s*xytb[0] + c*xytb[1] + ya;
    xytr[2] = ta + xytb[2];
}

static inline void TFN(s_xytcov_copy)(const TNAME xyta[3], const TNAME Ca[9],
                                      TNAME xytr[3], TNAME Cr[9])
{
    memcpy(xytr, xyta, 3 * sizeof(TNAME));
    memcpy(Cr, Ca, 9 * sizeof(TNAME));
}

static inline void TFN(s_xytcov_mul)(const TNAME xyta[3], const TNAME Ca[9],
                                      const TNAME xytb[3], const TNAME Cb[9],
                                      TNAME xytr[3], TNAME Cr[9])
{
    TNAME xa = xyta[0], ya = xyta[1], ta = xyta[2];
    TNAME xb = xytb[0], yb = xytb[1];

    TNAME sa = (TNAME)sin(ta), ca = (TNAME)cos(ta);

    TNAME P11 = Ca[0], P12 = Ca[1], P13 = Ca[2];
    TNAME              P22 = Ca[4], P23 = Ca[5];
    TNAME                           P33 = Ca[8];

    TNAME Q11 = Cb[0], Q12 = Cb[1], Q13 = Cb[2];
    TNAME              Q22 = Cb[4], Q23 = Cb[5];
    TNAME                           Q33 = Cb[8];

    TNAME JA13 = -sa*xb - ca*yb;
    TNAME JA23 = ca*xb - sa*yb;
    TNAME JB11 = ca;
    TNAME JB12 = -sa;
    TNAME JB21 = sa;
    TNAME JB22 = ca;

    Cr[0] = P33*JA13*JA13 + 2*P13*JA13 + Q11*JB11*JB11 + 2*Q12*JB11*JB12 + Q22*JB12*JB12 + P11;
    Cr[1] = P12 + JA23*(P13 + JA13*P33) + JA13*P23 + JB21*(JB11*Q11 + JB12*Q12) + JB22*(JB11*Q12 + JB12*Q22);
    Cr[2] = P13 + JA13*P33 + JB11*Q13 + JB12*Q23;
    Cr[3] = Cr[1];
    Cr[4] = P33*JA23*JA23 + 2*P23*JA23 + Q11*JB21*JB21 + 2*Q12*JB21*JB22 + Q22*JB22*JB22 + P22;
    Cr[5] = P23 + JA23*P33 + JB21*Q13 + JB22*Q23;
    Cr[6] = Cr[2];
    Cr[7] = Cr[5];
    Cr[8] = P33 + Q33;

    xytr[0] = ca*xb - sa*yb + xa;
    xytr[1] = sa*xb + ca*yb + ya;
    xytr[2] = xyta[2] + xytb[2];

/*
  // the code above is just an unrolling of the following:

        TNAME JA[][] = new TNAME[][] { { 1, 0, -sa*xb - ca*yb },
                                         { 0, 1, ca*xb - sa*yb },
                                         { 0, 0, 1 } };
        TNAME JB[][] = new TNAME[][] { { ca, -sa, 0 },
                                         { sa, ca, 0 },
                                         { 0,  0,  1 } };

        newge.P = LinAlg.add(LinAlg.matrixABCt(JA, P, JA),
                             LinAlg.matrixABCt(JB, ge.P, JB));
*/
}


static inline void TFN(s_xyt_inv)(const TNAME xyta[3], TNAME xytr[3])
{
    TNAME s = (TNAME)sin(xyta[2]), c = (TNAME)cos(xyta[2]);
    xytr[0] = -s*xyta[1] - c*xyta[0];
    xytr[1] = -c*xyta[1] + s*xyta[0];
    xytr[2] = -xyta[2];
}

static inline void TFN(s_xytcov_inv)(const TNAME xyta[3], const TNAME Ca[9],
                                      TNAME xytr[3], TNAME Cr[9])
{
    TNAME x = xyta[0], y = xyta[1], theta = xyta[2];
    TNAME s = (TNAME)sin(theta), c = (TNAME)cos(theta);

    TNAME J11 = -c, J12 = -s, J13 = -c*y + s*x;
    TNAME J21 = s,  J22 = -c, J23 = s*y + c*x;

    TNAME P11 = Ca[0], P12 = Ca[1], P13 = Ca[2];
    TNAME              P22 = Ca[4], P23 = Ca[5];
    TNAME                           P33 = Ca[8];

    Cr[0] = P11*J11*J11 + 2*P12*J11*J12 + 2*P13*J11*J13 +
        P22*J12*J12 + 2*P23*J12*J13 + P33*J13*J13;
    Cr[1] = J21*(J11*P11 + J12*P12 + J13*P13) +
        J22*(J11*P12 + J12*P22 + J13*P23) +
        J23*(J11*P13 + J12*P23 + J13*P33);
    Cr[2] = - J11*P13 - J12*P23 - J13*P33;
    Cr[3] = Cr[1];
    Cr[4] = P11*J21*J21 + 2*P12*J21*J22 + 2*P13*J21*J23 +
        P22*J22*J22 + 2*P23*J22*J23 + P33*J23*J23;
    Cr[5] = - J21*P13 - J22*P23 - J23*P33;
    Cr[6] = Cr[2];
    Cr[7] = Cr[5];
    Cr[8] = P33;

    /*
    // the code above is just an unrolling of the following:

    TNAME J[][] = new TNAME[][] { { -c, -s, -c*y + s*x },
                                    { s,  -c,  s*y + c*x },
                                    { 0,   0,     -1     } };
    ge.P = LinAlg.matrixABCt(J, P, J);
    */

    xytr[0] = -s*y - c*x;
    xytr[1] = -c*y + s*x;
    xytr[2] = -xyta[2];
}

// xytr = inv(xyta) * xytb
static inline void TFN(s_xyt_inv_mul)(const TNAME xyta[3], const TNAME xytb[3], TNAME xytr[3])
{
    TNAME theta = xyta[2];
    TNAME ca = (TNAME)cos(theta);
    TNAME sa = (TNAME)sin(theta);
    TNAME dx = xytb[0] - xyta[0];
    TNAME dy = xytb[1] - xyta[1];

    xytr[0] = ca*dx + sa*dy;
    xytr[1] = -sa*dx + ca*dy;
    xytr[2]= xytb[2] - xyta[2];
}

static inline void TFN(s_mat_add)(const TNAME *A, int Arows, int Acols,
                                   const TNAME *B, int Brows, int Bcols,
                                   TNAME *R, int Rrows, int Rcols)
{
    assert(Arows == Brows);
    assert(Arows == Rrows);
    assert(Bcols == Bcols);
    assert(Bcols == Rcols);

    for (int i = 0; i < Arows; i++)
        for (int j = 0; j < Bcols; j++)
            R[i*Acols + j] = A[i*Acols + j] + B[i*Acols + j];
}

// matrix should be in row-major order, allocated in a single packed
// array. (This is compatible with matd.)
static inline void TFN(s_mat_AB)(const TNAME *A, int Arows, int Acols,
                                  const TNAME *B, int Brows, int Bcols,
                                  TNAME *R, int Rrows, int Rcols)
{
    assert(Acols == Brows);
    assert(Rrows == Arows);
    assert(Bcols == Rcols);

    for (int Rrow = 0; Rrow < Rrows; Rrow++) {
        for (int Rcol = 0; Rcol < Rcols; Rcol++) {
            TNAME acc = 0;
            for (int i = 0; i < Acols; i++)
                acc += A[Rrow*Acols + i] * B[i*Bcols + Rcol];
            R[Rrow*Rcols + Rcol] = acc;
        }
    }
}

// matrix should be in row-major order, allocated in a single packed
// array. (This is compatible with matd.)
static inline void TFN(s_mat_ABt)(const TNAME *A, int Arows, int Acols,
                                  const TNAME *B, int Brows, int Bcols,
                                  TNAME *R, int Rrows, int Rcols)
{
    assert(Acols == Bcols);
    assert(Rrows == Arows);
    assert(Brows == Rcols);

    for (int Rrow = 0; Rrow < Rrows; Rrow++) {
        for (int Rcol = 0; Rcol < Rcols; Rcol++) {
            TNAME acc = 0;
            for (int i = 0; i < Acols; i++)
                acc += A[Rrow*Acols + i] * B[Rcol*Bcols + i];
            R[Rrow*Rcols + Rcol] = acc;
        }
    }
}

static inline void TFN(s_mat_ABC)(const TNAME *A, int Arows, int Acols,
                                  const TNAME *B, int Brows, int Bcols,
                                  const TNAME *C, int Crows, int Ccols,
                                  TNAME *R, int Rrows, int Rcols)
{
    TNAME *tmp = malloc(sizeof(TNAME)*Arows*Bcols);

    TFN(s_mat_AB)(A, Arows, Acols, B, Brows, Bcols, tmp, Arows, Bcols);
    TFN(s_mat_AB)(tmp, Arows, Bcols, C, Crows, Ccols, R, Rrows, Rcols);
    free(tmp);
}

static inline void TFN(s_mat_Ab)(const TNAME *A, int Arows, int Acols,
                                  const TNAME *B, int Blength,
                                  TNAME *R, int Rlength)
{
    assert(Acols == Blength);
    assert(Arows == Rlength);

    for (int Ridx = 0; Ridx < Rlength; Ridx++) {
        TNAME acc = 0;
        for (int i = 0; i < Blength; i++)
            acc += A[Ridx*Acols + i] * B[i];
        R[Ridx] = acc;
    }
}

static inline void TFN(s_mat_AtB)(const TNAME *A, int Arows, int Acols,
                                   const TNAME *B, int Brows, int Bcols,
                                   TNAME *R, int Rrows, int Rcols)
{
    assert(Arows == Brows);
    assert(Rrows == Acols);
    assert(Bcols == Rcols);

    for (int Rrow = 0; Rrow < Rrows; Rrow++) {
        for (int Rcol = 0; Rcol < Rcols; Rcol++) {
            TNAME acc = 0;
            for (int i = 0; i < Acols; i++)
                acc += A[i*Acols + Rrow] * B[i*Bcols + Rcol];
            R[Rrow*Rcols + Rcol] = acc;
        }
    }
}

static inline void TFN(s_quat_slerp)(const TNAME q0[4], const TNAME _q1[4], TNAME r[4], TNAME w)
{
    TNAME dot = TFN(s_dot)(q0, _q1, 4);

    TNAME q1[4];
    memcpy(q1, _q1, sizeof(TNAME) * 4);

    if (dot < 0) {
        // flip sign on one of them so we don't spin the "wrong
        // way" around. This doesn't change the rotation that the
        // quaternion represents.
        dot = -dot;
        for (int i = 0; i < 4; i++)
            q1[i] *= -1;
    }

    // if large dot product (1), slerp will scale both q0 and q1
    // by 0, and normalization will blow up.
    if (dot > 0.95) {

        for (int i = 0; i < 4; i++)
            r[i] = q0[i]*(1-w) + q1[i]*w;

    } else {
        TNAME angle = (TNAME)acos(dot);

        TNAME w0 = (TNAME)sin(angle*(1-w)), w1 = (TNAME)sin(angle*w);

        for (int i = 0; i < 4; i++)
            r[i] = q0[i]*w0 + q1[i]*w1;

        TFN(s_normalize)(r, 4, r);
    }
}

static inline void TFN(s_cross_product)(const TNAME v1[3], const TNAME v2[3], TNAME r[3])
{
    r[0] = v1[1]*v2[2] - v1[2]*v2[1];
    r[1] = v1[2]*v2[0] - v1[0]*v2[2];
    r[2] = v1[0]*v2[1] - v1[1]*v2[0];
}

////////////////////
static inline void TFN(s_mat44_identity)(TNAME out[16])
{
    memset(out, 0, 16 * sizeof(TNAME));
    out[0] = 1;
    out[5] = 1;
    out[10] = 1;
    out[15] = 1;
}

static inline void TFN(s_mat44_translate)(const TNAME txyz[3], TNAME out[16])
{
    TFN(s_mat44_identity)(out);

    for (int i = 0; i < 3; i++)
        out[4*i + 3] += txyz[i];
}

static inline void TFN(s_mat44_scale)(const TNAME sxyz[3], TNAME out[16])
{
    TFN(s_mat44_identity)(out);

    for (int i = 0; i < 3; i++)
        out[4*i + i] = sxyz[i];
}

static inline void TFN(s_mat44_rotate_z)(TNAME rad, TNAME out[16])
{
    TFN(s_mat44_identity)(out);
    TNAME s = (TNAME)sin(rad), c = (TNAME)cos(rad);
    out[0*4 + 0] = c;
    out[0*4 + 1] = -s;
    out[1*4 + 0] = s;
    out[1*4 + 1] = c;
}

static inline void TFN(s_mat44_rotate_y)(TNAME rad, TNAME out[16])
{
    TFN(s_mat44_identity)(out);
    TNAME s = (TNAME)sin(rad), c = (TNAME)cos(rad);
    out[0*4 + 0] = c;
    out[0*4 + 2] = s;
    out[2*4 + 0] = -s;
    out[2*4 + 2] = c;
}

static inline void TFN(s_mat44_rotate_x)(TNAME rad, TNAME out[16])
{
    TFN(s_mat44_identity)(out);
    TNAME s = (TNAME)sin(rad), c = (TNAME)cos(rad);
    out[1*4 + 1] = c;
    out[1*4 + 2] = -s;
    out[2*4 + 1] = s;
    out[2*4 + 2] = c;
}

// out = out * translate(txyz)
static inline void TFN(s_mat44_translate_self)(const TNAME txyz[3], TNAME out[16])
{
    TNAME tmp[16], prod[16];
    TFN(s_mat44_translate(txyz, tmp));
    TFN(s_mat_AB)(out, 4, 4, tmp, 4, 4, prod, 4, 4);
    memcpy(out, prod, sizeof(TNAME)*16);
}

static inline void TFN(s_mat44_scale_self)(const TNAME sxyz[3], TNAME out[16])
{
    TNAME tmp[16], prod[16];
    TFN(s_mat44_scale(sxyz, tmp));
    TFN(s_mat_AB)(out, 4, 4, tmp, 4, 4, prod, 4, 4);
    memcpy(out, prod, sizeof(TNAME)*16);
}

static inline void TFN(s_mat44_rotate_z_self)(TNAME rad, TNAME out[16])
{
    TNAME tmp[16], prod[16];
    TFN(s_mat44_rotate_z(rad, tmp));
    TFN(s_mat_AB)(out, 4, 4, tmp, 4, 4, prod, 4, 4);
    memcpy(out, prod, sizeof(TNAME)*16);
}

// out = inv(M)*in. Note: this assumes that mat44 is a rigid-body transformation.
static inline void TFN(s_mat44_inv)(const TNAME M[16], TNAME out[16])
{
// NB: M = T*R,  inv(M) = inv(R) * inv(T)

    // transpose of upper-left corner
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            out[4*i + j] = M[4*j + i];

    out[4*0 + 3] = 0;
    out[4*1 + 3] = 0;
    out[4*2 + 3] = 0;

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            out[4*i + 3] -= out[4*i + j] * M[4*j + 3];

    out[4*3 + 0] = 0;
    out[4*3 + 1] = 0;
    out[4*3 + 2] = 0;
    out[4*3 + 3] = 1;

/*    TNAME tmp[16];
    TFN(s_mat_AB)(M, 4, 4, out, 4, 4, tmp, 4, 4);
    printf("identity: ");
    TFN(s_print_mat)(tmp, 4, 4, "%15f"); */
}

// out = inv(M)*in
static inline void TFN(s_mat44_inv_transform_xyz)(const TNAME M[16], const TNAME in[3], TNAME out[3])
{
    TNAME T[16];
    TFN(s_mat44_inv)(M, T);

    TFN(s_mat44_transform_xyz)(T, in, out);
}

// out = (upper 3x3 of inv(M)) * in
static inline void TFN(s_mat44_inv_rotate_vector)(const TNAME M[16], const TNAME in[3], TNAME out[3])
{
    TNAME T[16];
    TFN(s_mat44_inv)(M, T);

    TFN(s_mat44_rotate_vector)(T, in, out);
}

static inline void TFN(s_elu_to_mat44)(const TNAME eye[3], const TNAME lookat[3], const TNAME _up[3],
                                       TNAME M[16])
{
    TNAME f[3];
    TFN(s_subtract)(lookat, eye, 3, f);
    TFN(s_normalize)(f, 3, f);

    TNAME up[3];

    // remove any component of 'up' that isn't perpendicular to the look direction.
    TFN(s_normalize)(_up, 3, up);

    TNAME up_dot = TFN(s_dot)(f, up, 3);
    for (int i = 0; i < 3; i++)
        up[i] -= up_dot*f[i];

    TFN(s_normalize_self)(up, 3);

    TNAME s[3], u[3];
    TFN(s_cross_product)(f, up, s);
    TFN(s_cross_product)(s, f, u);

    TNAME R[16] = {  s[0],  s[1],  s[2], 0,
                     u[0],  u[1],  u[2], 0,
                    -f[0], -f[1], -f[2], 0,
                     0,     0,     0,    1};

    TNAME T[16] = {1, 0, 0,  -eye[0],
                    0, 1, 0, -eye[1],
                    0, 0, 1, -eye[2],
                    0, 0, 0, 1};

    // M is the extrinsics matrix [R | t] where t = -R*c
    TNAME tmp[16];
    TFN(s_mat_AB)(R, 4, 4, T, 4, 4, tmp, 4, 4);
    TFN(s_mat44_inv)(tmp, M);
}

// Computes the cholesky factorization of A, putting the lower
// triangular matrix into R.
static inline void TFN(s_mat33_chol)(const TNAME *A, int Arows, int Acols,
                                     TNAME *R, int Brows, int Bcols)
{
    assert(Arows == Brows);
    assert(Bcols == Bcols);

    // A[0] = R[0]*R[0]
    R[0] = (TNAME)sqrt(A[0]);

    // A[1] = R[0]*R[3];
    R[3] = A[1] / R[0];

    // A[2] = R[0]*R[6];
    R[6] = A[2] / R[0];

    // A[4] = R[3]*R[3] + R[4]*R[4]
    R[4] = (TNAME)sqrt(A[4] - R[3]*R[3]);

    // A[5] = R[3]*R[6] + R[4]*R[7]
    R[7] = (A[5] - R[3]*R[6]) / R[4];

    // A[8] = R[6]*R[6] + R[7]*R[7] + R[8]*R[8]
    R[8] = (TNAME)sqrt(A[8] - R[6]*R[6] - R[7]*R[7]);

    R[1] = 0;
    R[2] = 0;
    R[5] = 0;
}

static inline void TFN(s_mat33_lower_tri_inv)(const TNAME *A, int Arows, int Acols,
                                              TNAME *R, int Rrows, int Rcols)
{
    // A[0]*R[0] = 1
    R[0] = 1 / A[0];

    // A[3]*R[0] + A[4]*R[3] = 0
    R[3] = -A[3]*R[0] / A[4];

    // A[4]*R[4] = 1
    R[4] = 1 / A[4];

    // A[6]*R[0] + A[7]*R[3] + A[8]*R[6] = 0
    R[6] = (-A[6]*R[0] - A[7]*R[3]) / A[8];

    // A[7]*R[4] + A[8]*R[7] = 0
    R[7] = -A[7]*R[4] / A[8];

    // A[8]*R[8] = 1
    R[8] = 1 / A[8];
}


static inline void TFN(s_mat33_sym_solve)(const TNAME *A, int Arows, int Acols,
                                          const TNAME *B, int Brows, int Bcols,
                                          TNAME *R, int Rrows, int Rcols)
{
    assert(Arows == Acols);
    assert(Acols == 3);
    assert(Brows == 3);
    assert(Bcols == 1);
    assert(Rrows == 3);
    assert(Rcols == 1);

    TNAME L[9];
    TFN(s_mat33_chol)(A, 3, 3, L, 3, 3);

    TNAME M[9];
    TFN(s_mat33_lower_tri_inv)(L, 3, 3, M, 3, 3);

    double tmp[3];
    tmp[0] = M[0]*B[0];
    tmp[1] = M[3]*B[0] + M[4]*B[1];
    tmp[2] = M[6]*B[0] + M[7]*B[1] + M[8]*B[2];

    R[0] = (TNAME)(M[0]*tmp[0] + M[3]*tmp[1] + M[6]*tmp[2]);
    R[1] = (TNAME)(M[4]*tmp[1] + M[7]*tmp[2]);
    R[2] = (TNAME)(M[8]*tmp[2]);
}

/*
// solve Ax = B. Assumes A is symmetric; uses cholesky factorization
static inline void TFN(s_mat_solve_chol)(const TNAME *A, int Arows, int Acols,
                                         const TNAME *B, int Brows, int Bcols,
                                         TNAME *R, int Rrows, int Rcols)
{
    assert(Arows == Acols);
    assert(Arows == Brows);
    assert(Acols == Rrows);
    assert(Bcols == Rcols);

    //
}
*/
#undef TRRFN
#undef TRFN
#undef TFN
