// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv { namespace ximgproc {

    static inline Moments operator& ( const Moments & lhs, const Matx22d & rhs )
    {
        return Moments (
            lhs.m00,
            rhs ( 0, 0 ) * lhs.m10 + rhs ( 0, 1 ) * lhs.m01,
            rhs ( 1, 0 ) * lhs.m10 + rhs ( 1, 1 ) * lhs.m01,
            rhs ( 0, 0 ) * rhs ( 0, 0 ) * lhs.m20 + rhs ( 0, 1 ) * rhs ( 0, 1 ) * lhs.m02 + 2 * rhs ( 0, 0 ) * rhs ( 0, 1 ) * lhs.m11,
            rhs ( 0, 0 ) * rhs ( 1, 0 ) * lhs.m20 + rhs ( 0, 1 ) * rhs ( 1, 1 ) * lhs.m02 + ( rhs ( 0, 0 ) * rhs ( 1, 1 ) + rhs ( 0, 1 ) * rhs ( 1, 0 ) ) * lhs.m11,
            rhs ( 1, 0 ) * rhs ( 1, 0 ) * lhs.m20 + rhs ( 1, 1 ) * rhs ( 1, 1 ) * lhs.m02 + 2 * rhs ( 1, 0 ) * rhs ( 1, 1 ) * lhs.m11,
            rhs ( 0, 0 ) * rhs ( 0, 0 ) * rhs ( 0, 0 ) * lhs.m30 + 3 * rhs ( 0, 0 ) * rhs ( 0, 0 ) * rhs ( 0, 1 ) * lhs.m21 + 3 * rhs ( 0, 0 ) * rhs ( 0, 1 ) * rhs ( 0, 1 ) * lhs.m12 + rhs ( 0, 1 ) * rhs ( 0, 1 ) * rhs ( 0, 1 ) * lhs.m03,
            rhs ( 0, 0 ) * rhs ( 0, 0 ) * rhs ( 1, 0 ) * lhs.m30 + ( rhs ( 0, 0 ) * rhs ( 0, 0 ) * rhs ( 1, 1 ) + 2 * rhs ( 0, 0 ) * rhs ( 0, 1 ) * rhs ( 1, 0 ) ) * lhs.m21 + ( 2 * rhs ( 0, 0 ) * rhs ( 0, 1 ) * rhs ( 1, 1 ) + rhs ( 0, 1 ) * rhs ( 0, 1 ) * rhs ( 1, 0 ) ) * lhs.m12 + rhs ( 0, 1 ) * rhs ( 0, 1 ) * rhs ( 1, 1 ) * lhs.m03,
            rhs ( 0, 0 ) * rhs ( 1, 0 ) * rhs ( 1, 0 ) * lhs.m30 + ( rhs ( 1, 0 ) * rhs ( 1, 0 ) * rhs ( 0, 1 ) + 2 * rhs ( 0, 0 ) * rhs ( 1, 0 ) * rhs ( 1, 1 ) ) * lhs.m21 + ( 2 * rhs ( 0, 1 ) * rhs ( 1, 0 ) * rhs ( 1, 1 ) + rhs ( 1, 1 ) * rhs ( 1, 1 ) * rhs ( 0, 0 ) ) * lhs.m12 + rhs ( 0, 1 ) * rhs ( 1, 1 ) * rhs ( 1, 1 ) * lhs.m03,
            rhs ( 1, 0 ) * rhs ( 1, 0 ) * rhs ( 1, 0 ) * lhs.m30 + 3 * rhs ( 1, 0 ) * rhs ( 1, 0 ) * rhs ( 1, 1 ) * lhs.m21 + 3 * rhs ( 1, 0 ) * rhs ( 1, 1 ) * rhs ( 1, 1 ) * lhs.m12 + rhs ( 1, 1 ) * rhs ( 1, 1 ) * rhs ( 1, 1 ) * lhs.m03
        );
    }

    static inline Matx23d operator| ( const Matx22d & lhs, const Matx21d & rhs )
    {
        return Matx23d ( lhs ( 0, 0 ), lhs ( 0, 1 ), rhs ( 0 ), lhs ( 1, 0 ), lhs ( 1, 1 ), rhs ( 1 ) );
    }

    Matx23d PeiLinNormalization ( InputArray I )
    {
        const Moments  M = moments ( I );
        const double  l1 = ( M.mu20 / M.m00 + M.mu02 / M.m00 + sqrt ( ( M.mu20 / M.m00 - M.mu02 / M.m00 ) * ( M.mu20 / M.m00 - M.mu02 / M.m00 ) + 4 * M.mu11 / M.m00 * M.mu11 / M.m00 ) ) / 2;
        const double  l2 = ( M.mu20 / M.m00 + M.mu02 / M.m00 - sqrt ( ( M.mu20 / M.m00 - M.mu02 / M.m00 ) * ( M.mu20 / M.m00 - M.mu02 / M.m00 ) + 4 * M.mu11 / M.m00 * M.mu11 / M.m00 ) ) / 2;
        const double  ex = ( M.mu11 / M.m00 ) / sqrt ( ( l1 - M.mu20 / M.m00 ) * ( l1 - M.mu20 / M.m00 ) + M.mu11 / M.m00 * M.mu11 / M.m00 );
        const double  ey = ( l1 - M.mu20 / M.m00 ) / sqrt ( ( l1 - M.mu20 / M.m00 ) * ( l1 - M.mu20 / M.m00 ) + M.mu11 / M.m00 * M.mu11 / M.m00 );
        const Matx22d  E = Matx22d ( ex, ey, -ey, ex );
        const double   p = min ( I.size().height, I.size().width ) / 8;
        const Matx22d  W = Matx22d ( p / sqrt ( l1 ), 0, 0, p / sqrt ( l2 ) );
        const Matx21d  c = Matx21d ( M.m10 / M.m00, M.m01 / M.m00 );
        const Matx21d  i = Matx21d ( I.size().width / 2, I.size().height / 2 );
        const Moments  N = M & W * E;
        const double  t1 = N.mu12 + N.mu30;
        const double  t2 = N.mu03 + N.mu21;
        const double phi = atan2 ( -t1, t2 );
        const double psi = ( -t1 * sin ( phi ) + t2 * cos ( phi ) >= 0 ) ? phi : ( phi + CV_PI  );
        const Matx22d  A = Matx22d ( cos ( psi ), sin ( psi ), -sin ( psi ), cos ( psi ) );
        return ( A * W * E ) | ( i - A * W * E * c );
    }

    void PeiLinNormalization ( InputArray I, OutputArray T )
    {
        T.assign ( Mat ( PeiLinNormalization ( I ) ) );
    }

}} // namespace
