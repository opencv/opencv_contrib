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

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

#include "common/doubles.h"

/** SVD 2x2.

    Computes singular values and vectors without squaring the input
    matrix. With double precision math, results are accurate to about
    1E-16.

    U = [ cos(theta) -sin(theta) ]
        [ sin(theta)  cos(theta) ]

    S = [ e  0 ]
        [ 0  f ]

    V = [ cos(phi)   -sin(phi) ]
        [ sin(phi)   cos(phi)  ]


    Our strategy is basically to analytically multiply everything out
    and then rearrange so that we can solve for theta, phi, e, and
    f. (Derivation by ebolson@umich.edu 5/2016)

   V' = [ CP  SP ]
        [ -SP CP ]

USV' = [ CT -ST ][  e*CP  e*SP ]
       [ ST  CT ][ -f*SP  f*CP ]

     = [e*CT*CP + f*ST*SP     e*CT*SP - f*ST*CP ]
       [e*ST*CP - f*SP*CT     e*SP*ST + f*CP*CT ]

A00+A11 = e*CT*CP + f*ST*SP + e*SP*ST + f*CP*CT
        = e*(CP*CT + SP*ST) + f*(SP*ST + CP*CT)
    	= (e+f)(CP*CT + SP*ST)
B0	    = (e+f)*cos(P-T)

A00-A11 = e*CT*CP + f*ST*SP - e*SP*ST - f*CP*CT
        = e*(CP*CT - SP*ST) - f*(-ST*SP + CP*CT)
	    = (e-f)(CP*CT - SP*ST)
B1	    = (e-f)*cos(P+T)

A01+A10 = e*CT*SP - f*ST*CP + e*ST*CP - f*SP*CT
	    = e(CT*SP + ST*CP) - f*(ST*CP + SP*CT)
	    = (e-f)*(CT*SP + ST*CP)
B2	    = (e-f)*sin(P+T)

A01-A10 = e*CT*SP - f*ST*CP - e*ST*CP + f*SP*CT
	= e*(CT*SP - ST*CP) + f(SP*CT - ST*CP)
	= (e+f)*(CT*SP - ST*CP)
B3	= (e+f)*sin(P-T)

B0 = (e+f)*cos(P-T)
B1 = (e-f)*cos(P+T)
B2 = (e-f)*sin(P+T)
B3 = (e+f)*sin(P-T)

B3/B0 = tan(P-T)

B2/B1 = tan(P+T)
 **/
void svd22(const double A[4], double U[4], double S[2], double V[4])
{
    double A00 = A[0];
    double A01 = A[1];
    double A10 = A[2];
    double A11 = A[3];

    double B0 = A00 + A11;
    double B1 = A00 - A11;
    double B2 = A01 + A10;
    double B3 = A01 - A10;

    double PminusT = atan2(B3, B0);
    double PplusT = atan2(B2, B1);

    double P = (PminusT + PplusT) / 2;
    double T = (-PminusT + PplusT) / 2;

    double CP = cos(P), SP = sin(P);
    double CT = cos(T), ST = sin(T);

    U[0] = CT;
    U[1] = -ST;
    U[2] = ST;
    U[3] = CT;

    V[0] = CP;
    V[1] = -SP;
    V[2] = SP;
    V[3] = CP;

    // C0 = e+f. There are two ways to compute C0; we pick the one
    // that is better conditioned.
    double CPmT = cos(P-T), SPmT = sin(P-T);
    double C0 = 0;
    if (fabs(CPmT) > fabs(SPmT))
        C0 = B0 / CPmT;
    else
        C0 = B3 / SPmT;

    // C1 = e-f. There are two ways to compute C1; we pick the one
    // that is better conditioned.
    double CPpT = cos(P+T), SPpT = sin(P+T);
    double C1 = 0;
    if (fabs(CPpT) > fabs(SPpT))
        C1 = B1 / CPpT;
    else
        C1 = B2 / SPpT;

    // e and f are the singular values
    double e = (C0 + C1) / 2;
    double f = (C0 - C1) / 2;

    if (e < 0) {
        e = -e;
        U[0] = -U[0];
        U[2] = -U[2];
    }

    if (f < 0) {
        f = -f;
        U[1] = -U[1];
        U[3] = -U[3];
    }

    // sort singular values.
    if (e > f) {
        // already in big-to-small order.
        S[0] = e;
        S[1] = f;
    } else {
        // Curiously, this code never seems to get invoked.  Why is it
        // that S[0] always ends up the dominant vector?  However,
        // this code has been tested (flipping the logic forces us to
        // sort the singular values in ascending order).
        //
        // P = [ 0 1 ; 1 0 ]
        // USV' = (UP)(PSP)(PV')
        //      = (UP)(PSP)(VP)'
        //      = (UP)(PSP)(P'V')'
        S[0] = f;
        S[1] = e;

        // exchange columns of U and V
        double tmp[2];
        tmp[0] = U[0];
        tmp[1] = U[2];
        U[0] = U[1];
        U[2] = U[3];
        U[1] = tmp[0];
        U[3] = tmp[1];

        tmp[0] = V[0];
        tmp[1] = V[2];
        V[0] = V[1];
        V[2] = V[3];
        V[1] = tmp[0];
        V[3] = tmp[1];
    }

    /*
    double SM[4] = { S[0], 0, 0, S[1] };

    doubles_print_mat(U, 2, 2, "%20.10g");
    doubles_print_mat(SM, 2, 2, "%20.10g");
    doubles_print_mat(V, 2, 2, "%20.10g");
    printf("A:\n");
    doubles_print_mat(A, 2, 2, "%20.10g");

    double SVt[4];
    doubles_mat_ABt(SM, 2, 2, V, 2, 2, SVt, 2, 2);
    double USVt[4];
    doubles_mat_AB(U, 2, 2, SVt, 2, 2, USVt, 2, 2);

    printf("USVt\n");
    doubles_print_mat(USVt, 2, 2, "%20.10g");

    double diff[4];
    for (int i = 0; i < 4; i++)
        diff[i] = A[i] - USVt[i];

    printf("diff\n");
    doubles_print_mat(diff, 2, 2, "%20.10g");

    */

}


// for the matrix [a b; b d]
void svd_sym_singular_values(double A00, double A01, double A11,
                             double *Lmin, double *Lmax)
{
    double A10 = A01;

    double B0 = A00 + A11;
    double B1 = A00 - A11;
    double B2 = A01 + A10;
    double B3 = A01 - A10;

    double PminusT = atan2(B3, B0);
    double PplusT = atan2(B2, B1);

    double P = (PminusT + PplusT) / 2;
    double T = (-PminusT + PplusT) / 2;

    // C0 = e+f. There are two ways to compute C0; we pick the one
    // that is better conditioned.
    double CPmT = cos(P-T), SPmT = sin(P-T);
    double C0 = 0;
    if (fabs(CPmT) > fabs(SPmT))
        C0 = B0 / CPmT;
    else
        C0 = B3 / SPmT;

    // C1 = e-f. There are two ways to compute C1; we pick the one
    // that is better conditioned.
    double CPpT = cos(P+T), SPpT = sin(P+T);
    double C1 = 0;
    if (fabs(CPpT) > fabs(SPpT))
        C1 = B1 / CPpT;
    else
        C1 = B2 / SPpT;

    // e and f are the singular values
    double e = (C0 + C1) / 2;
    double f = (C0 - C1) / 2;

    *Lmin = fmin(e, f);
    *Lmax = fmax(e, f);
}
