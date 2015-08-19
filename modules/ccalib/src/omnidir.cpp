/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2015, Baisheng Lai (laibaisheng@gmail.com), Zhejiang University,
// all rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/**
 * This module was accepted as a GSoC 2015 project for OpenCV, authored by
 * Baisheng Lai, mentored by Bo Li.
 *
 * The omnidirectional camera in this module is denoted by the catadioptric
 * model. Please refer to Mei's paper for details of the camera model:
 *
 *      C. Mei and P. Rives, "Single view point omnidirectional camera
 *      calibration from planar grids", in ICRA 2007.
 *
 * The implementation of the calibration part is based on Li's calibration
 * toolbox:
 *
 *     B. Li, L. Heng, K. Kevin  and M. Pollefeys, "A Multiple-Camera System
 *     Calibration Toolbox Using A Feature Descriptor-Based Calibration
 *     Pattern", in IROS 2013.
 */
#include "precomp.hpp"
#include "opencv2/ccalib/omnidir.hpp"
#include <fstream>
#include <iostream>
namespace cv { namespace
{
    struct JacobianRow
    {
        Matx13d dom,dT;
        Matx12d df;
        double ds;
        Matx12d dc;
        double dxi;
        Matx14d dkp;    // distortion k1,k2,p1,p2
    };
}}

/////////////////////////////////////////////////////////////////////////////
//////// projectPoints
void cv::omnidir::projectPoints(InputArray objectPoints, OutputArray imagePoints,
                InputArray rvec, InputArray tvec, InputArray K, double xi, InputArray D, OutputArray jacobian)
{

    CV_Assert(objectPoints.type() == CV_64FC3 || objectPoints.type() == CV_32FC3);
    CV_Assert((rvec.depth() == CV_64F || rvec.depth() == CV_32F) && rvec.total() == 3);
    CV_Assert((tvec.depth() == CV_64F || tvec.depth() == CV_32F) && tvec.total() == 3);
    CV_Assert((K.type() == CV_64F || K.type() == CV_32F) && K.size() == Size(3,3));
    CV_Assert((D.type() == CV_64F || D.type() == CV_32F) && D.total() == 4);

    imagePoints.create(objectPoints.size(), CV_MAKETYPE(objectPoints.depth(), 2));

    int n = (int)objectPoints.total();

    Vec3d om = rvec.depth() == CV_32F ? (Vec3d)*rvec.getMat().ptr<Vec3f>() : *rvec.getMat().ptr<Vec3d>();
    Vec3d T  = tvec.depth() == CV_32F ? (Vec3d)*tvec.getMat().ptr<Vec3f>() : *tvec.getMat().ptr<Vec3d>();

    Vec2d f,c;
    double s;
    if (K.depth() == CV_32F)
    {
        Matx33f Kc = K.getMat();
        f = Vec2f(Kc(0,0), Kc(1,1));
        c = Vec2f(Kc(0,2),Kc(1,2));
        s = (double)Kc(0,1);
    }
    else
    {
        Matx33d Kc = K.getMat();
        f = Vec2d(Kc(0,0), Kc(1,1));
        c = Vec2d(Kc(0,2),Kc(1,2));
        s = Kc(0,1);
    }

    Vec4d kp = D.depth() == CV_32F ? (Vec4d)*D.getMat().ptr<Vec4f>() : *D.getMat().ptr<Vec4d>();
    //Vec<double, 4> kp= (Vec<double,4>)*D.getMat().ptr<Vec<double,4> >();

    const Vec3d* Xw_alld = objectPoints.getMat().ptr<Vec3d>();
    const Vec3f* Xw_allf = objectPoints.getMat().ptr<Vec3f>();
    Vec2d* xpd = imagePoints.getMat().ptr<Vec2d>();
    Vec2f* xpf = imagePoints.getMat().ptr<Vec2f>();

    Matx33d R;
    Matx<double, 3, 9> dRdom;
    Rodrigues(om, R, dRdom);

    JacobianRow *Jn = 0;
    if (jacobian.needed())
    {
        int nvars = 2+2+1+4+3+3+1; // f,c,s,kp,om,T,xi
        jacobian.create(2*int(n), nvars, CV_64F);
        Jn = jacobian.getMat().ptr<JacobianRow>(0);
    }

    double k1=kp[0],k2=kp[1];
    double p1 = kp[2], p2 = kp[3];

    for (int i = 0; i < n; i++)
    {
        // convert to camera coordinate
        Vec3d Xw = objectPoints.depth() == CV_32F ? (Vec3d)Xw_allf[i] : Xw_alld[i];

        Vec3d Xc = (Vec3d)(R*Xw + T);

        // convert to unit sphere
        Vec3d Xs = Xc/cv::norm(Xc);

        // convert to normalized image plane
        Vec2d xu = Vec2d(Xs[0]/(Xs[2]+xi), Xs[1]/(Xs[2]+xi));

        // add distortion
        Vec2d xd;
        double r2 = xu[0]*xu[0]+xu[1]*xu[1];
        double r4 = r2*r2;

        xd[0] = xu[0]*(1+k1*r2+k2*r4) + 2*p1*xu[0]*xu[1] + p2*(r2+2*xu[0]*xu[0]);
        xd[1] = xu[1]*(1+k1*r2+k2*r4) + p1*(r2+2*xu[1]*xu[1]) + 2*p2*xu[0]*xu[1];

        // convert to pixel coordinate
        Vec2d final;
        final[0] = f[0]*xd[0]+s*xd[1]+c[0];
        final[1] = f[1]*xd[1]+c[1];

        if (objectPoints.depth() == CV_32F)
        {
            xpf[i] = final;
        }
        else
        {
            xpd[i] = final;
        }
        /*xpd[i][0] = f[0]*xd[0]+s*xd[1]+c[0];
        xpd[i][1] = f[1]*xd[1]+c[1];*/

        if (jacobian.needed())
        {
            double dXcdR_a[] = {Xw[0],Xw[1],Xw[2],0,0,0,0,0,0,
                                0,0,0,Xw[0],Xw[1],Xw[2],0,0,0,
                                0,0,0,0,0,0,Xw[0],Xw[1],Xw[2]};
            Matx<double,3, 9> dXcdR(dXcdR_a);
            Matx33d dXcdom = dXcdR * dRdom.t();
            double r_1 = 1.0/norm(Xc);
            double r_3 = pow(r_1,3);
            Matx33d dXsdXc(r_1-Xc[0]*Xc[0]*r_3, -(Xc[0]*Xc[1])*r_3, -(Xc[0]*Xc[2])*r_3,
                           -(Xc[0]*Xc[1])*r_3, r_1-Xc[1]*Xc[1]*r_3, -(Xc[1]*Xc[2])*r_3,
                           -(Xc[0]*Xc[2])*r_3, -(Xc[1]*Xc[2])*r_3, r_1-Xc[2]*Xc[2]*r_3);
            Matx23d dxudXs(1/(Xs[2]+xi),    0,    -Xs[0]/(Xs[2]+xi)/(Xs[2]+xi),
                           0,    1/(Xs[2]+xi),    -Xs[1]/(Xs[2]+xi)/(Xs[2]+xi));
            // pre-compute some reusable things
            double temp1 = 2*k1*xu[0] + 4*k2*xu[0]*r2;
            double temp2 = 2*k1*xu[1] + 4*k2*xu[1]*r2;
            Matx22d dxddxu(k2*r4+6*p2*xu[0]+2*p1*xu[1]+xu[0]*temp1+k1*r2+1,    2*p1*xu[0]+2*p2*xu[1]+xu[0]*temp2,
                           2*p1*xu[0]+2*p2*xu[1]+xu[1]*temp1,    k2*r4+2*p2*xu[0]+6*p1*xu[1]+xu[1]*temp2+k1*r2+1);
            Matx22d dxpddxd(f[0], s,
                            0, f[1]);
            Matx23d dxpddXc = dxpddxd * dxddxu * dxudXs * dXsdXc;

            // derivative of xpd respect to om
            Matx23d dxpddom = dxpddXc * dXcdom;
            Matx33d dXcdT(1.0,0.0,0.0,
                          0.0,1.0,0.0,
                          0.0,0.0,1.0);
            // derivative of xpd respect to T

            Matx23d dxpddT = dxpddXc * dXcdT;
            Matx21d dxudxi(-Xs[0]/(Xs[2]+xi)/(Xs[2]+xi),
                           -Xs[1]/(Xs[2]+xi)/(Xs[2]+xi));

            // derivative of xpd respect to xi
            Matx21d dxpddxi = dxpddxd * dxddxu * dxudxi;
            Matx<double,2,4> dxddkp(xu[0]*r2, xu[0]*r4, 2*xu[0]*xu[1], r2+2*xu[0]*xu[0],
                                    xu[1]*r2, xu[1]*r4, r2+2*xu[1]*xu[1], 2*xu[0]*xu[1]);

            // derivative of xpd respect to kp
            Matx<double,2,4> dxpddkp = dxpddxd * dxddkp;

            // derivative of xpd respect to f
            Matx22d dxpddf(xd[0], 0,
                           0, xd[1]);

            // derivative of xpd respect to c
            Matx22d dxpddc(1, 0,
                           0, 1);

            Jn[0].dom = dxpddom.row(0);
            Jn[1].dom = dxpddom.row(1);
            Jn[0].dT = dxpddT.row(0);
            Jn[1].dT = dxpddT.row(1);
            Jn[0].dkp = dxpddkp.row(0);
            Jn[1].dkp = dxpddkp.row(1);
            Jn[0].dxi = dxpddxi(0,0);
            Jn[1].dxi = dxpddxi(1,0);
            Jn[0].df = dxpddf.row(0);
            Jn[1].df = dxpddf.row(1);
            Jn[0].dc = dxpddc.row(0);
            Jn[1].dc = dxpddc.row(1);
            Jn[0].ds = xd[1];
            Jn[1].ds = 0;
            Jn += 2;
         }
    }
}

/////////////////////////////////////////////////////////////////////////////
//////// undistortPoints
void cv::omnidir::undistortPoints( InputArray distorted, OutputArray undistorted,
    InputArray K, InputArray D, InputArray xi, InputArray R)
{
    CV_Assert(distorted.type() == CV_64FC2 || distorted.type() == CV_32FC2);
    CV_Assert(R.empty() || (!R.empty() && (R.size() == Size(3, 3) || R.total() * R.channels() == 3)
        && (R.depth() == CV_64F || R.depth() == CV_32F)));
    CV_Assert((D.depth() == CV_64F || D.depth() == CV_32F) && D.total() == 4);
    CV_Assert(K.size() == Size(3, 3) && (K.depth() == CV_64F || K.depth() == CV_32F));
    CV_Assert(xi.total() == 1 && (xi.depth() == CV_64F || xi.depth() == CV_32F));

    undistorted.create(distorted.size(), distorted.type());

    cv::Vec2d f, c;
    double s = 0.0;
    if (K.depth() == CV_32F)
    {
        Matx33f camMat = K.getMat();
        f = Vec2f(camMat(0,0), camMat(1,1));
        c = Vec2f(camMat(0,2), camMat(1,2));
        s = (double)camMat(0,1);
    }
    else if (K.depth() == CV_64F)
    {
        Matx33d camMat = K.getMat();
        f = Vec2d(camMat(0,0), camMat(1,1));
        c = Vec2d(camMat(0,2), camMat(1,2));
        s = camMat(0,1);
    }

    Vec4d kp = D.depth()==CV_32F ? (Vec4d)*D.getMat().ptr<Vec4f>():(Vec4d)*D.getMat().ptr<Vec4d>();
    Vec2d k = Vec2d(kp[0], kp[1]);
    Vec2d p = Vec2d(kp[2], kp[3]);

    double _xi = xi.depth() == CV_32F ? (double)*xi.getMat().ptr<float>() : *xi.getMat().ptr<double>();
    cv::Matx33d RR = cv::Matx33d::eye();
    // R is om
    if(!R.empty() && R.total()*R.channels() == 3)
    {
        cv::Vec3d rvec;
        R.getMat().convertTo(rvec, CV_64F);
        cv::Rodrigues(rvec, RR);
    }
    else if (!R.empty() && R.size() == Size(3,3))
    {
        R.getMat().convertTo(RR, CV_64F);
    }

    const cv::Vec2d *srcd = distorted.getMat().ptr<cv::Vec2d>();
    const cv::Vec2f *srcf = distorted.getMat().ptr<cv::Vec2f>();

    cv::Vec2d *dstd = undistorted.getMat().ptr<cv::Vec2d>();
    cv::Vec2f *dstf = undistorted.getMat().ptr<cv::Vec2f>();

    int n = (int)distorted.total();
    for (int i = 0; i < n; i++)
    {
        Vec2d pi = distorted.depth() == CV_32F ? (Vec2d)srcf[i]:(Vec2d)srcd[i];    // image point
        Vec2d pp((pi[0]*f[1]-c[0]*f[1]-s*(pi[1]-c[1]))/(f[0]*f[1]), (pi[1]-c[1])/f[1]); //plane
        Vec2d pu = pp;    // points without distortion

        // remove distortion iteratively
        for (int j = 0; j < 20; j++)
        {
            double r2 = pu[0]*pu[0] + pu[1]*pu[1];
            double r4 = r2*r2;
            pu[0] = (pp[0] - 2*p[0]*pu[0]*pu[1] - p[1]*(r2+2*pu[0]*pu[0])) / (1 + k[0]*r2 + k[1]*r4);
            pu[1] = (pp[1] - 2*p[1]*pu[0]*pu[1] - p[0]*(r2+2*pu[1]*pu[1])) / (1 + k[0]*r2 + k[1]*r4);
        }

        // project to unit sphere
        double r2 = pu[0]*pu[0] + pu[1]*pu[1];
        double a = (r2 + 1);
        double b = 2*_xi*r2;
        double cc = r2*_xi*_xi-1;
        double Zs = (-b + sqrt(b*b - 4*a*cc))/(2*a);
        Vec3d Xw = Vec3d(pu[0]*(Zs + _xi), pu[1]*(Zs +_xi), Zs);

        // rotate
        Xw = RR * Xw;

        // project back to sphere
        Vec3d Xs = Xw / cv::norm(Xw);

        // reproject to camera plane
        Vec3d ppu = Vec3d(Xs[0]/(Xs[2]+_xi), Xs[1]/(Xs[2]+_xi), 1.0);
        if (undistorted.depth() == CV_32F)
        {
            dstf[i] = Vec2f((float)ppu[0], (float)ppu[1]);
        }
        else if (undistorted.depth() == CV_64F)
        {
            dstd[i] = Vec2d(ppu[0], ppu[1]);
        }
    }
}


/////////////////////////////////////////////////////////////////////////////
//////// cv::omnidir::initUndistortRectifyMap
void cv::omnidir::initUndistortRectifyMap(InputArray K, InputArray D, InputArray xi, InputArray R, InputArray P,
    const cv::Size& size, int m1type, OutputArray map1, OutputArray map2, int flags)
{
    CV_Assert( m1type == CV_16SC2 || m1type == CV_32F || m1type <=0 );
    map1.create( size, m1type <= 0 ? CV_16SC2 : m1type );
    map2.create( size, map1.type() == CV_16SC2 ? CV_16UC1 : CV_32F );

    CV_Assert((K.depth() == CV_32F || K.depth() == CV_64F) && (D.depth() == CV_32F || D.depth() == CV_64F));
    CV_Assert(K.size() == Size(3, 3) && (D.empty() || D.total() == 4));
    CV_Assert(P.empty()|| (P.depth() == CV_32F || P.depth() == CV_64F));
    CV_Assert(P.empty() || P.size() == Size(3, 3) || P.size() == Size(4, 3));
    CV_Assert(R.empty() || (R.depth() == CV_32F || R.depth() == CV_64F));
    CV_Assert(R.empty() || R.size() == Size(3, 3) || R.total() * R.channels() == 3);
    CV_Assert(flags == RECTIFY_PERSPECTIVE || flags == RECTIFY_CYLINDRICAL || flags == RECTIFY_LONGLATI
        || flags == RECTIFY_STEREOGRAPHIC);
    CV_Assert(xi.total() == 1 && (xi.depth() == CV_32F || xi.depth() == CV_64F));

    cv::Vec2d f, c;
    double s;
    if (K.depth() == CV_32F)
    {
        Matx33f camMat = K.getMat();
        f = Vec2f(camMat(0, 0), camMat(1, 1));
        c = Vec2f(camMat(0, 2), camMat(1, 2));
        s = (double)camMat(0,1);
    }
    else
    {
        Matx33d camMat = K.getMat();
        f = Vec2d(camMat(0, 0), camMat(1, 1));
        c = Vec2d(camMat(0, 2), camMat(1, 2));
        s = camMat(0,1);
    }

    Vec4d kp = Vec4d::all(0);
    if (!D.empty())
        kp = D.depth() == CV_32F ? (Vec4d)*D.getMat().ptr<Vec4f>(): *D.getMat().ptr<Vec4d>();
    double _xi = xi.depth() == CV_32F ? (double)*xi.getMat().ptr<float>() : *xi.getMat().ptr<double>();
    Vec2d k = Vec2d(kp[0], kp[1]);
    Vec2d p = Vec2d(kp[2], kp[3]);
    cv::Matx33d RR  = cv::Matx33d::eye();
    if (!R.empty() && R.total() * R.channels() == 3)
    {
        cv::Vec3d rvec;
        R.getMat().convertTo(rvec, CV_64F);
        cv::Rodrigues(rvec, RR);
    }
    else if (!R.empty() && R.size() == Size(3, 3))
        R.getMat().convertTo(RR, CV_64F);

    cv::Matx33d PP = cv::Matx33d::eye();
    if (!P.empty())
        P.getMat().colRange(0, 3).convertTo(PP, CV_64F);
    else
        PP = K.getMat();

    cv::Matx33d iKR = (PP*RR).inv(cv::DECOMP_SVD);
    cv::Matx33d iK = PP.inv(cv::DECOMP_SVD);
    cv::Matx33d iR = RR.inv(cv::DECOMP_SVD);

    if (flags == omnidir::RECTIFY_PERSPECTIVE)
    {
        for (int i = 0; i < size.height; ++i)
        {
            float* m1f = map1.getMat().ptr<float>(i);
            float* m2f = map2.getMat().ptr<float>(i);
            short*  m1 = (short*)m1f;
            ushort* m2 = (ushort*)m2f;

            double _x = i*iKR(0, 1) + iKR(0, 2),
                   _y = i*iKR(1, 1) + iKR(1, 2),
                   _w = i*iKR(2, 1) + iKR(2, 2);
            for(int j = 0; j < size.width; ++j, _x+=iKR(0,0), _y+=iKR(1,0), _w+=iKR(2,0))
            {
                // project back to unit sphere
                double r = sqrt(_x*_x + _y*_y + _w*_w);
                double Xs = _x / r;
                double Ys = _y / r;
                double Zs = _w / r;
                // project to image plane
                double xu = Xs / (Zs + _xi),
                    yu = Ys / (Zs + _xi);
                // add distortion
                double r2 = xu*xu + yu*yu;
                double r4 = r2*r2;
                double xd = (1+k[0]*r2+k[1]*r4)*xu + 2*p[0]*xu*yu + p[1]*(r2+2*xu*xu);
                double yd = (1+k[0]*r2+k[1]*r4)*yu + p[0]*(r2+2*yu*yu) + 2*p[1]*xu*yu;
                // to image pixel
                double u = f[0]*xd + s*yd + c[0];
                double v = f[1]*yd + c[1];

                if( m1type == CV_16SC2 )
                {
                    int iu = cv::saturate_cast<int>(u*cv::INTER_TAB_SIZE);
                    int iv = cv::saturate_cast<int>(v*cv::INTER_TAB_SIZE);
                    m1[j*2+0] = (short)(iu >> cv::INTER_BITS);
                    m1[j*2+1] = (short)(iv >> cv::INTER_BITS);
                    m2[j] = (ushort)((iv & (cv::INTER_TAB_SIZE-1))*cv::INTER_TAB_SIZE + (iu & (cv::INTER_TAB_SIZE-1)));
                }
                else if( m1type == CV_32FC1 )
                {
                    m1f[j] = (float)u;
                    m2f[j] = (float)v;
                }
            }
        }
    }
    else if(flags == omnidir::RECTIFY_CYLINDRICAL || flags == omnidir::RECTIFY_LONGLATI ||
        flags == omnidir::RECTIFY_STEREOGRAPHIC)
    {
        for (int i = 0; i < size.height; ++i)
        {
            float* m1f = map1.getMat().ptr<float>(i);
            float* m2f = map2.getMat().ptr<float>(i);
            short*  m1 = (short*)m1f;
            ushort* m2 = (ushort*)m2f;

            // for RECTIFY_LONGLATI, theta and h are longittude and latitude
            double theta = i*iK(0, 1) + iK(0, 2),
                   h     = i*iK(1, 1) + iK(1, 2);

            for (int j = 0; j < size.width; ++j, theta+=iK(0,0), h+=iK(1,0))
            {
                double _xt = 0.0, _yt = 0.0, _wt = 0.0;
                if (flags == omnidir::RECTIFY_CYLINDRICAL)
                {
                    //_xt = std::sin(theta);
                    //_yt = h;
                    //_wt = std::cos(theta);
                    _xt = std::cos(theta);
                    _yt = std::sin(theta);
                    _wt = h;
                }
                else if (flags == omnidir::RECTIFY_LONGLATI)
                {
                    _xt = -std::cos(theta);
                    _yt = -std::sin(theta) * std::cos(h);
                    _wt = std::sin(theta) * std::sin(h);
                }
                else if (flags == omnidir::RECTIFY_STEREOGRAPHIC)
                {
                    double a = theta*theta + h*h + 4;
                    double b = -2*theta*theta - 2*h*h;
                    double c2 = theta*theta + h*h -4;

                    _yt = (-b-std::sqrt(b*b - 4*a*c2))/(2*a);
                    _xt = theta*(1 - _yt) / 2;
                    _wt = h*(1 - _yt) / 2;
                }
                double _x = iR(0,0)*_xt + iR(0,1)*_yt + iR(0,2)*_wt;
                double _y = iR(1,0)*_xt + iR(1,1)*_yt + iR(1,2)*_wt;
                double _w = iR(2,0)*_xt + iR(2,1)*_yt + iR(2,2)*_wt;

                double r = sqrt(_x*_x + _y*_y + _w*_w);
                double Xs = _x / r;
                double Ys = _y / r;
                double Zs = _w / r;
                // project to image plane
                double xu = Xs / (Zs + _xi),
                       yu = Ys / (Zs + _xi);
                // add distortion
                double r2 = xu*xu + yu*yu;
                double r4 = r2*r2;
                double xd = (1+k[0]*r2+k[1]*r4)*xu + 2*p[0]*xu*yu + p[1]*(r2+2*xu*xu);
                double yd = (1+k[0]*r2+k[1]*r4)*yu + p[0]*(r2+2*yu*yu) + 2*p[1]*xu*yu;
                // to image pixel
                double u = f[0]*xd + s*yd + c[0];
                double v = f[1]*yd + c[1];

                if( m1type == CV_16SC2 )
                {
                    int iu = cv::saturate_cast<int>(u*cv::INTER_TAB_SIZE);
                    int iv = cv::saturate_cast<int>(v*cv::INTER_TAB_SIZE);
                    m1[j*2+0] = (short)(iu >> cv::INTER_BITS);
                    m1[j*2+1] = (short)(iv >> cv::INTER_BITS);
                    m2[j] = (ushort)((iv & (cv::INTER_TAB_SIZE-1))*cv::INTER_TAB_SIZE + (iu & (cv::INTER_TAB_SIZE-1)));
                }
                else if( m1type == CV_32FC1 )
                {
                    m1f[j] = (float)u;
                    m2f[j] = (float)v;
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::omnidir::undistortImage

void cv::omnidir::undistortImage(InputArray distorted, OutputArray undistorted,
    InputArray K, InputArray D, InputArray xi, int flags, InputArray Knew, const Size& new_size, InputArray R)
{
    Size size = new_size.area() != 0 ? new_size : distorted.size();

    cv::Mat map1, map2;
    omnidir::initUndistortRectifyMap(K, D, xi, R, Knew, size, CV_16SC2, map1, map2, flags);
    cv::remap(distorted, undistorted, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::omnidir::internal::initializeCalibration

void cv::omnidir::internal::initializeCalibration(InputArrayOfArrays patternPoints, InputArrayOfArrays imagePoints, Size size,
    OutputArrayOfArrays omAll, OutputArrayOfArrays tAll, OutputArray K, double& xi, OutputArray idx)
{
    // For details please refer to Section III from Li's IROS 2013 paper

    double u0 = size.width / 2;
    double v0 = size.height / 2;

    int n_img = (int)imagePoints.total();

    std::vector<cv::Vec3d> v_omAll(n_img), v_tAll(n_img);

    std::vector<double> gammaAll(n_img);

    K.create(3, 3, CV_64F);
    Mat _K;
    for (int image_idx = 0; image_idx < n_img; ++image_idx)
    {
        cv::Mat objPoints, imgPoints;
		patternPoints.getMat(image_idx).copyTo(objPoints);
		imagePoints.getMat(image_idx).copyTo(imgPoints);

		int n_point = imgPoints.rows * imgPoints.cols;
		if (objPoints.rows != n_point)
			objPoints = objPoints.reshape(3, n_point);
		if (imgPoints.rows != n_point)
			imgPoints = imgPoints.reshape(2, n_point);

        // objectPoints should be 3-channel data, imagePoints should be 2-channel data
        CV_Assert(objPoints.type() == CV_64FC3 && imgPoints.type() == CV_64FC2 );

        std::vector<cv::Mat> xy, uv;
        cv::split(objPoints, xy);
        cv::split(imgPoints, uv);


        cv::Mat x = xy[0].reshape(1, n_point), y = xy[1].reshape(1, n_point),
                u = uv[0].reshape(1, n_point) - u0, v = uv[1].reshape(1, n_point) - v0;

        cv::Mat sqrRho = u.mul(u) + v.mul(v);
        // compute extrinsic parameters
        cv::Mat M(n_point, 6, CV_64F);
        Mat(-v.mul(x)).copyTo(M.col(0));
        Mat(-v.mul(y)).copyTo(M.col(1));
        Mat(u.mul(x)).copyTo(M.col(2));
        Mat(u.mul(y)).copyTo(M.col(3));
        Mat(-v).copyTo(M.col(4));
        Mat(u).copyTo(M.col(5));

        Mat W,U,V;
        cv::SVD::compute(M, W, U, V,SVD::FULL_UV);
        V = V.t();

        double miniReprojectError = 1e5;
        // the signs of r1, r2, r3 are unknown, so they can be flipped.
        for (int coef = 1; coef >= -1; coef-=2)
        {
            double r11 = V.at<double>(0, 5) * coef;
            double r12 = V.at<double>(1, 5) * coef;
            double r21 = V.at<double>(2, 5) * coef;
            double r22 = V.at<double>(3, 5) * coef;
            double t1 = V.at<double>(4, 5) * coef;
            double t2 = V.at<double>(5, 5) * coef;

            Mat roots;
            double r31s;
            solvePoly(Matx13d(-(r11*r12+r21*r22)*(r11*r12+r21*r22), r11*r11+r21*r21-r12*r12-r22*r22, 1), roots);

            if (roots.at<Vec2d>(0)[0] > 0)
                r31s = sqrt(roots.at<Vec2d>(0)[0]);
            else
                r31s = sqrt(roots.at<Vec2d>(1)[0]);

            for (int coef2 = 1; coef2 >= -1; coef2-=2)
            {
                double r31 = r31s * coef2;
                double r32 = -(r11*r12 + r21*r22) / r31;

                cv::Vec3d r1(r11, r21, r31);
                cv::Vec3d r2(r12, r22, r32);
                cv::Vec3d t(t1, t2, 0);
                double scale = 1 / cv::norm(r1);
                r1 = r1 * scale;
                r2 = r2 * scale;
                t = t * scale;

                // compute intrisic parameters
                // Form equations in Scaramuzza's paper
                // A Toolbox for Easily Calibrating Omnidirectional Cameras
                Mat A(n_point*2, 3, CV_64F);
                Mat((r1[1]*x + r2[1]*y + t[1])/2).copyTo(A.rowRange(0, n_point).col(0));
                Mat((r1[0]*x + r2[0]*y + t[0])/2).copyTo(A.rowRange(n_point, 2*n_point).col(0));
                Mat(-A.col(0).rowRange(0, n_point).mul(sqrRho)).copyTo(A.col(1).rowRange(0, n_point));
                Mat(-A.col(0).rowRange(n_point, 2*n_point).mul(sqrRho)).copyTo(A.col(1).rowRange(n_point, 2*n_point));
                Mat(-v).copyTo(A.rowRange(0, n_point).col(2));
                Mat(-u).copyTo(A.rowRange(n_point, 2*n_point).col(2));

                // Operation to avoid bad numerical-condition of A
                Vec3d maxA, minA;
                for (int j = 0; j < A.cols; j++)
                {
                    cv::minMaxLoc(cv::abs(A.col(j)), &minA[j], &maxA[j]);
                    A.col(j) = A.col(j) / maxA[j];
                }

                Mat B(n_point*2 , 1, CV_64F);
                Mat(v.mul(r1[2]*x + r2[2]*y)).copyTo(B.rowRange(0, n_point));
                Mat(u.mul(r1[2]*x + r2[2]*y)).copyTo(B.rowRange(n_point, 2*n_point));

                Mat res = A.inv(DECOMP_SVD) * B;
                res = res.mul(1/Mat(maxA));

                double gamma = sqrt(res.at<double>(0) / res.at<double>(1));
                t[2] = res.at<double>(2);

                cv::Vec3d r3 = r1.cross(r2);

                Matx33d R(r1[0], r2[0], r3[0],
                          r1[1], r2[1], r3[1],
                          r1[2], r2[2], r3[2]);
                Vec3d om;
                Rodrigues(R, om);

                // project pattern points to images
                Mat projedImgPoints;
                Matx33d Kc(gamma, 0, u0, 0, gamma, v0, 0, 0, 1);

                // reproject error
                cv::omnidir::projectPoints(objPoints, projedImgPoints, om, t, Kc, 1, Matx14d(0, 0, 0, 0), cv::noArray());
                double reprojectError = omnidir::internal::computeMeanReproErr(imgPoints, projedImgPoints);

                // if this reproject error is smaller
                if (reprojectError < miniReprojectError)
                {
                    miniReprojectError = reprojectError;
                    v_omAll[image_idx] = om;
                    v_tAll[image_idx] = t;
                    gammaAll[image_idx] = gamma;
                }
            }
        }
    }

    // filter initial results whose reproject errors are too large
    std::vector<double> reProjErrorFilter,v_gammaFilter;
    std::vector<Vec3d> omFilter, tFilter;
    double gammaFinal = 0;

    // choose median value
    size_t n = gammaAll.size() / 2;
    std::nth_element(gammaAll.begin(), gammaAll.begin()+n, gammaAll.end());
    gammaFinal = gammaAll[n];

    _K = Mat(Matx33d(gammaFinal, 0, u0, 0, gammaFinal, v0, 0, 0, 1));
    _K.convertTo(K, CV_64F);
    std::vector<int> _idx;
    // recompute reproject error using the final gamma
    for (int i = 0; i< n_img; i++)
    {
        Mat _projected;
        cv::omnidir::projectPoints(patternPoints.getMat(i), _projected, v_omAll[i], v_tAll[i], _K, 1, Matx14d(0, 0, 0, 0), cv::noArray());
        double _error = omnidir::internal::computeMeanReproErr(imagePoints.getMat(i), _projected);
        if(_error < 100)
        {
            _idx.push_back(i);
            omFilter.push_back(v_omAll[i]);
            tFilter.push_back(v_tAll[i]);
        }
    }

    if (idx.needed())
    {
        idx.create(1, (int)_idx.size(), CV_32S);
        Mat idx_m = idx.getMat();
        for (int j = 0; j < (int)idx_m.total(); j++)
        {
            idx_m.at<int>(j) = _idx[j];
        }
    }

    if(omAll.kind() == _InputArray::STD_VECTOR_MAT)
    {
        for (int i = 0; i < (int)omFilter.size(); ++i)
        {
            omAll.getMat(i) = Mat(omFilter[i]);
            tAll.getMat(i) = Mat(tFilter[i]);
        }
    }
    else
    {
        cv::Mat(omFilter).convertTo(omAll, CV_64FC3);
        cv::Mat(tFilter).convertTo(tAll, CV_64FC3);
    }
    xi = 1;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::omnidir::internal::initializeStereoCalibration

void cv::omnidir::internal::initializeStereoCalibration(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
    const Size& size1, const Size& size2, OutputArray om, OutputArray T, OutputArrayOfArrays omL, OutputArrayOfArrays tL, OutputArray K1, OutputArray D1, OutputArray K2, OutputArray D2,
    double &xi1, double &xi2, int flags, OutputArray idx)
{
    Mat idx1, idx2;
    Matx33d _K1, _K2;
    Matx14d _D1, _D2;
    Mat _xi1m, _xi2m;

    std::vector<Vec3d> omAllTemp1, omAllTemp2, tAllTemp1, tAllTemp2;

    omnidir::calibrate(objectPoints, imagePoints1, size1, _K1, _xi1m, _D1, omAllTemp1, tAllTemp1, flags, TermCriteria(3, 100, 1e-6), idx1);
    omnidir::calibrate(objectPoints, imagePoints2, size2, _K2, _xi2m, _D2, omAllTemp2, tAllTemp2, flags, TermCriteria(3, 100, 1e-6), idx2);

    // find the intersection idx
    Mat interIdx1, interIdx2, interOri;

    getInterset(idx1, idx2, interIdx1, interIdx2, interOri);
    if (idx.empty())
        idx.create(1, (int)interOri.total(), CV_32S);
    interOri.copyTo(idx.getMat());

    int n_inter = (int)interIdx1.total();

    std::vector<Vec3d> omAll1(n_inter), omAll2(n_inter), tAll1(n_inter), tAll2(n_inter);
    for(int i = 0; i < (int)interIdx1.total(); ++i)
    {
        omAll1[i] = omAllTemp1[interIdx1.at<int>(i)];
        tAll1[i] = tAllTemp1[interIdx1.at<int>(i)];
        omAll2[i] = omAllTemp2[interIdx2.at<int>(i)];
        tAll2[i] = tAllTemp2[interIdx2.at<int>(i)];
    }

    // initialize R,T
    Mat omEstAll(1, n_inter, CV_64FC3), tEstAll(1, n_inter, CV_64FC3);
    Mat R1, R2, T1, T2, omLR, TLR, RLR;
    for (int i = 0; i < n_inter; ++i)
    {
        Rodrigues(omAll1[i], R1);
        Rodrigues(omAll2[i], R2);
        T1 = Mat(tAll1[i]).reshape(1, 3);
        T2 = Mat(tAll2[i]).reshape(1, 3);
        RLR = R2 * R1.t();
        TLR = T2 - RLR*T1;
        Rodrigues(RLR, omLR);
        omLR.reshape(3, 1).copyTo(omEstAll.col(i));
        TLR.reshape(3, 1).copyTo(tEstAll.col(i));
    }
    Vec3d omEst = internal::findMedian3(omEstAll);
    Vec3d tEst = internal::findMedian3(tEstAll);

    Mat(omEst).copyTo(om.getMat());
    Mat(tEst).copyTo(T.getMat());

    if (omL.empty())
    {
        omL.create((int)omAll1.size(), 1, CV_64FC3);
    }
    if (tL.empty())
    {
        tL.create((int)tAll1.size(), 1, CV_64FC3);
    }

    if(omL.kind() == _InputArray::STD_VECTOR_MAT)
    {
        for(int i = 0; i < n_inter; ++i)
        {
            omL.create(3, 1, CV_64F, i, true);
            tL.create(3, 1, CV_64F, i, true);
            omL.getMat(i) = Mat(omAll1[i]);
            tL.getMat(i) = Mat(tAll1[i]);
        }
    }
    else
    {
        cv::Mat(omAll1).convertTo(omL, CV_64FC3);
        cv::Mat(tAll1).convertTo(tL, CV_64FC3);
    }
    if (K1.empty())
    {
        K1.create(3, 3, CV_64F);
        K2.create(3, 3, CV_64F);
    }
    if (D1.empty())
    {
        D1.create(1, 4, CV_64F);
        D2.create(1, 4, CV_64F);
    }
    Mat(_K1).copyTo(K1.getMat());
    Mat(_K2).copyTo(K2.getMat());

    Mat(_D1).copyTo(D1.getMat());
    Mat(_D2).copyTo(D2.getMat());

    xi1 = _xi1m.at<double>(0);
    xi2 = _xi2m.at<double>(0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::omnidir::internal::computeJacobian

void cv::omnidir::internal::computeJacobian(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints,
    InputArray parameters, Mat& JTJ_inv, Mat& JTE, int flags, double epsilon)
{
    CV_Assert(!objectPoints.empty() && objectPoints.type() == CV_64FC3);
    CV_Assert(!imagePoints.empty() && imagePoints.type() == CV_64FC2);

    int n = (int)objectPoints.total();

    Mat JTJ = Mat::zeros(10 + 6*n, 10 + 6*n, CV_64F);
    JTJ_inv = Mat::zeros(10 + 6*n, 10 + 6*n, CV_64F);
    JTE = Mat::zeros(10 + 6*n, 1, CV_64F);

    int nPointsAll = 0;
    for (int i = 0; i < n; ++i)
    {
        nPointsAll += (int)objectPoints.getMat(i).total();
    }

    Mat J = Mat::zeros(2*nPointsAll, 10+6*n, CV_64F);
    Mat exAll = Mat::zeros(2*nPointsAll, 10+6*n, CV_64F);
    double *para = parameters.getMat().ptr<double>();
    Matx33d K(para[6*n], para[6*n+2], para[6*n+3],
        0,    para[6*n+1], para[6*n+4],
        0,    0,  1);
    Matx14d D(para[6*n+6], para[6*n+7], para[6*n+8], para[6*n+9]);
    double xi = para[6*n+5];
    for (int i = 0; i < n; i++)
    {
        Mat objPoints, imgPoints, om, T;
		objectPoints.getMat(i).copyTo(objPoints);
		imagePoints.getMat(i).copyTo(imgPoints);
		objPoints = objPoints.reshape(3, objPoints.rows*objPoints.cols);
		imgPoints = imgPoints.reshape(2, imgPoints.rows*imgPoints.cols);

        om = parameters.getMat().colRange(i*6, i*6+3);
        T = parameters.getMat().colRange(i*6+3, (i+1)*6);
        Mat imgProj, jacobian;
        omnidir::projectPoints(objPoints, imgProj, om, T, K, xi, D, jacobian);
        Mat projError = imgPoints - imgProj;

        // The intrinsic part of Jacobian
        Mat JIn(jacobian.rows, 10, CV_64F);
        Mat JEx(jacobian.rows, 6, CV_64F);

        jacobian.colRange(6, 16).copyTo(JIn);
        jacobian.colRange(0, 6).copyTo(JEx);

        JTJ(Rect(6*n, 6*n, 10, 10)) = JTJ(Rect(6*n, 6*n, 10, 10)) + JIn.t()*JIn;

        JTJ(Rect(i*6, i*6, 6, 6)) = JEx.t() * JEx;

        Mat JExTIn = JEx.t() * JIn;

        JExTIn.copyTo(JTJ(Rect(6*n, i*6, 10, 6)));

        Mat(JIn.t()*JEx).copyTo(JTJ(Rect(i*6, 6*n, 6, 10)));

        JTE(Rect(0, 6*n, 1, 10)) = JTE(Rect(0, 6*n,1, 10)) + JIn.t() * projError.reshape(1, 2*(int)projError.total());
        JTE(Rect(0, i*6, 1, 6)) = JEx.t() * projError.reshape(1, 2*(int)projError.total());

        //int nPoints = objectPoints.getMat(i).total();
        //JIn.copyTo(J(Rect(6*n, i*nPoints*2, 10, nPoints*2)));
        //JEx.copyTo(J(Rect(6*i, i*nPoints*2, 6, nPoints*2)));
        //projError.reshape(1, 2*projError.rows).copyTo(exAll.rowRange(i*2*nPoints, (i+1)*2*nPoints));
    }
    //JTJ = J.t()*J;
    //JTE = J.t()*exAll;
    std::vector<int> _idx(6*n+10, 1);
    flags2idx(flags, _idx, n);

    subMatrix(JTJ, JTJ, _idx, _idx);
    subMatrix(JTE, JTE, std::vector<int>(1, 1), _idx);
    // in case JTJ is singular
	//SVD svd(JTJ, SVD::NO_UV);
	//double cond = svd.w.at<double>(0)/svd.w.at<double>(5);

	//if (cond_JTJ.needed())
	//{
	//	cond_JTJ.create(1, 1, CV_64F);
	//	cond_JTJ.getMat().at<double>(0) = cond;
	//}

    //double epsilon = 1e-4*std::exp(cond);
    JTJ_inv = Mat(JTJ+epsilon).inv();
}

void cv::omnidir::internal::computeJacobianStereo(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
    InputArray parameters, Mat& JTJ_inv, Mat& JTE, int flags, double epsilon)
{
    CV_Assert(!objectPoints.empty() && objectPoints.type() == CV_64FC3);
    CV_Assert(!imagePoints1.empty() && imagePoints1.type() == CV_64FC2);
    CV_Assert(!imagePoints2.empty() && imagePoints2.type() == CV_64FC2);
    CV_Assert((imagePoints1.total() == imagePoints2.total()) && (imagePoints1.total() == objectPoints.total()));

    // compute Jacobian matrix by naive way
    int n_img = (int)objectPoints.total();
    int n_points = (int)objectPoints.getMat(0).total();
    Mat J = Mat::zeros(4 * n_points * n_img, 20 + 6 * (n_img + 1), CV_64F);
    Mat exAll = Mat::zeros(4 * n_points * n_img, 1, CV_64F);
    double *para = parameters.getMat().ptr<double>();
    int offset1 = (n_img + 1) * 6;
    int offset2 = offset1 + 10;
    Matx33d K1(para[offset1], para[offset1+2], para[offset1+3],
        0,    para[offset1+1], para[offset1+4],
        0,    0,  1);
    Matx14d D1(para[offset1+6], para[offset1+7], para[offset1+8], para[offset1+9]);
    double xi1 = para[offset1+5];

    Matx33d K2(para[offset2], para[offset2+2], para[offset2+3],
        0,    para[offset2+1], para[offset2+4],
        0,    0,  1);
    Matx14d D2(para[offset2+6], para[offset2+7], para[offset2+8], para[offset2+9]);
    double xi2 = para[offset2+5];

    Mat om = parameters.getMat().reshape(1, 1).colRange(0, 3);
    Mat T = parameters.getMat().reshape(1, 1).colRange(3, 6);

    for (int i = 0; i < n_img; i++)
    {
        Mat objPointsi, imgPoints1i, imgPoints2i, om1, T1;
        objectPoints.getMat(i).copyTo(objPointsi);
        imagePoints1.getMat(i).copyTo(imgPoints1i);
        imagePoints2.getMat(i).copyTo(imgPoints2i);
        objPointsi = objPointsi.reshape(3, objPointsi.rows*objPointsi.cols);
        imgPoints1i = imgPoints1i.reshape(2, imgPoints1i.rows*imgPoints1i.cols);
        imgPoints2i = imgPoints2i.reshape(2, imgPoints2i.rows*imgPoints2i.cols);

        om1 = parameters.getMat().reshape(1, 1).colRange((1 + i) * 6, (1 + i) * 6 + 3);
        T1 = parameters.getMat().reshape(1, 1).colRange((1 + i) * 6 + 3, (i + 1) * 6 + 6);


        Mat imgProj1, imgProj2, jacobian1, jacobian2;

        // jacobian for left image
        cv::omnidir::projectPoints(objPointsi, imgProj1, om1, T1, K1, xi1, D1, jacobian1);
        Mat projError1 = imgPoints1i - imgProj1;
        //Mat JIn1(jacobian1.rows, 10, CV_64F);
        //Mat JEx1(jacobian1.rows, 6, CV_64F);
        jacobian1.colRange(6, 16).copyTo(J(Rect(6*(n_img+1), i*n_points*4, 10, n_points*2)));
        jacobian1.colRange(0, 6).copyTo(J(Rect(6+i*6, i*n_points*4, 6, n_points*2)));
        projError1.reshape(1, 2*n_points).copyTo(exAll.rowRange(i*4*n_points, (i*4+2)*n_points));

        //jacobian for right image
        Mat om2, T2, dom2dom1, dom2dT1, dom2dom, dom2dT, dT2dom1, dT2dT1, dT2dom, dT2dT;
        cv::omnidir::internal::compose_motion(om1, T1, om, T, om2, T2, dom2dom1, dom2dT1, dom2dom, dom2dT, dT2dom1, dT2dT1, dT2dom, dT2dT);
        cv::omnidir::projectPoints(objPointsi, imgProj2, om2, T2, K2, xi2, D2, jacobian2);
        Mat projError2 = imgPoints2i - imgProj2;
        projError2.reshape(1, 2*n_points).copyTo(exAll.rowRange((i*4+2)*n_points, (i*4+4)*n_points));
        Mat dxrdom = jacobian2.colRange(0, 3) * dom2dom + jacobian2.colRange(3, 6) * dT2dom;
        Mat dxrdT = jacobian2.colRange(0, 3) * dom2dT + jacobian2.colRange(3, 6) * dT2dT;
        Mat dxrdom1 = jacobian2.colRange(0, 3) * dom2dom1 + jacobian2.colRange(3, 6) * dT2dom1;
        Mat dxrdT1 = jacobian2.colRange(0, 3) * dom2dT1 + jacobian2.colRange(3, 6) * dT2dT1;

        dxrdom.copyTo(J(Rect(0, (i*4+2)*n_points, 3, n_points*2)));
        dxrdT.copyTo(J(Rect(3, (i*4+2)*n_points, 3, n_points*2)));
        dxrdom1.copyTo(J(Rect(6+i*6, (i*4+2)*n_points, 3, n_points*2)));
        dxrdT1.copyTo(J(Rect(6+i*6+3, (i*4+2)*n_points, 3, n_points*2)));
        jacobian2.colRange(6, 16).copyTo(J(Rect(6*(n_img+1)+10, (4*i+2)*n_points, 10, n_points*2)));
    }

    std::vector<int> _idx(6*(n_img+1)+20, 1);
    flags2idxStereo(flags, _idx, n_img);

    Mat JTJ = J.t()*J;
    JTE = J.t()*exAll;
    subMatrix(JTJ, JTJ, _idx, _idx);
    subMatrix(JTE, JTE, std::vector<int>(1, 1), _idx);

    JTJ_inv = Mat(JTJ+epsilon).inv();
}

// This function is from fisheye.cpp
void cv::omnidir::internal::compose_motion(InputArray _om1, InputArray _T1, InputArray _om2, InputArray _T2, Mat& om3, Mat& T3, Mat& dom3dom1,
    Mat& dom3dT1, Mat& dom3dom2, Mat& dom3dT2, Mat& dT3dom1, Mat& dT3dT1, Mat& dT3dom2, Mat& dT3dT2)
{
    Mat om1 = _om1.getMat();
    Mat om2 = _om2.getMat();
    Mat T1 = _T1.getMat().reshape(1, 3);
    Mat T2 = _T2.getMat().reshape(1, 3);

    //% Rotations:
    Mat R1, R2, R3, dR1dom1(9, 3, CV_64FC1), dR2dom2;
    Rodrigues(om1, R1, dR1dom1);
    Rodrigues(om2, R2, dR2dom2);
    //JRodriguesMatlab(dR1dom1, dR1dom1);
    //JRodriguesMatlab(dR2dom2, dR2dom2);
    dR1dom1 = dR1dom1.t();
    dR2dom2 = dR2dom2.t();

    R3 = R2 * R1;
    Mat dR3dR2, dR3dR1;
    //dAB(R2, R1, dR3dR2, dR3dR1);
    matMulDeriv(R2, R1, dR3dR2, dR3dR1);

    Mat dom3dR3;
    Rodrigues(R3, om3, dom3dR3);
    //JRodriguesMatlab(dom3dR3, dom3dR3);
    dom3dR3 = dom3dR3.t();
    dom3dom1 = dom3dR3 * dR3dR1 * dR1dom1;
    dom3dom2 = dom3dR3 * dR3dR2 * dR2dom2;
    dom3dT1 = Mat::zeros(3, 3, CV_64FC1);
    dom3dT2 = Mat::zeros(3, 3, CV_64FC1);

    //% Translations:
    Mat T3t = R2 * T1;
    Mat dT3tdR2, dT3tdT1;
    //dAB(R2, T1, dT3tdR2, dT3tdT1);
    matMulDeriv(R2, T1, dT3tdR2, dT3tdT1);
    Mat dT3tdom2 = dT3tdR2 * dR2dom2;
    T3 = T3t + T2;
    dT3dT1 = dT3tdT1;
    dT3dT2 = Mat::eye(3, 3, CV_64FC1);
    dT3dom2 = dT3tdom2;
    dT3dom1 = Mat::zeros(3, 3, CV_64FC1);
}

double cv::omnidir::calibrate(InputArray patternPoints, InputArray imagePoints, Size size,
    InputOutputArray K, InputOutputArray xi, InputOutputArray D, OutputArrayOfArrays omAll, OutputArrayOfArrays tAll,
    int flags, TermCriteria criteria, OutputArray idx)
{
    CV_Assert(!patternPoints.empty() && !imagePoints.empty() && patternPoints.total() == imagePoints.total());
    CV_Assert((patternPoints.type() == CV_64FC3 && imagePoints.type() == CV_64FC2) ||
        (patternPoints.type() == CV_32FC3 && imagePoints.type() == CV_32FC2));
    CV_Assert(patternPoints.getMat(0).channels() == 3 && imagePoints.getMat(0).channels() == 2);
    CV_Assert((!K.empty() && K.size() == Size(3,3)) || K.empty());
    CV_Assert((!D.empty() && D.total() == 4) || D.empty());
    CV_Assert((!xi.empty() && xi.total() == 1) || xi.empty());
    CV_Assert((!omAll.empty() && omAll.depth() == patternPoints.depth()) || omAll.empty());
    CV_Assert((!tAll.empty() && tAll.depth() == patternPoints.depth()) || tAll.empty());
    int depth = patternPoints.depth();

    std::vector<Mat> _patternPoints, _imagePoints;

    for (int i = 0; i < (int)patternPoints.total(); ++i)
    {
        _patternPoints.push_back(patternPoints.getMat(i));
        _imagePoints.push_back(imagePoints.getMat(i));
        if (depth == CV_32F)
        {
            _patternPoints[i].convertTo(_patternPoints[i], CV_64FC3);
            _imagePoints[i].convertTo(_imagePoints[i], CV_64FC2);
        }
    }

    double _xi;
    // initialization
    std::vector<Vec3d> _omAll, _tAll;
    Matx33d _K;
    Matx14d _D;
    Mat _idx;
    cv::omnidir::internal::initializeCalibration(_patternPoints, _imagePoints, size, _omAll, _tAll, _K, _xi, _idx);
    std::vector<Mat> _patternPointsTmp = _patternPoints;
    std::vector<Mat> _imagePointsTmp = _imagePoints;

    _patternPoints.clear();
    _imagePoints.clear();
    // erase
    for (int i = 0; i < (int)_idx.total(); i++)
    {
        _patternPoints.push_back(_patternPointsTmp[_idx.at<int>(i)]);
        _imagePoints.push_back(_imagePointsTmp[_idx.at<int>(i)]);
    }

    int n = (int)_patternPoints.size();
    Mat finalParam(1, 10 + 6*n, CV_64F);
    Mat currentParam(1, 10 + 6*n, CV_64F);
    cv::omnidir::internal::encodeParameters(_K, _omAll, _tAll, Mat::zeros(1,4,CV_64F), _xi, currentParam);

    // optimization
    const double alpha_smooth = 0.01;
    //const double thresh_cond = 1e6;
    double change = 1;
    for(int iter = 0; ; ++iter)
    {
        if ((criteria.type == 1 && iter >= criteria.maxCount)  ||
            (criteria.type == 2 && change <= criteria.epsilon) ||
            (criteria.type == 3 && (change <= criteria.epsilon || iter >= criteria.maxCount)))
            break;
        double alpha_smooth2 = 1 - std::pow(1 - alpha_smooth, (double)iter + 1.0);
        Mat JTJ_inv, JTError;
		double epsilon = 0.01 * std::pow(0.9, (double)iter/10);
        cv::omnidir::internal::computeJacobian(_patternPoints, _imagePoints, currentParam, JTJ_inv, JTError, flags, epsilon);

        // Gauss¨CNewton
        Mat G = alpha_smooth2*JTJ_inv * JTError;

        omnidir::internal::fillFixed(G, flags, n);

        finalParam = currentParam + G.t();

        change = norm(G) / norm(currentParam);

        currentParam = finalParam.clone();

        cv::omnidir::internal::decodeParameters(currentParam, _K, _omAll, _tAll, _D, _xi);
        //double repr = internal::computeMeanReproErr(_patternPoints, _imagePoints, _K, _D, _xi, _omAll, _tAll);
    }
    cv::omnidir::internal::decodeParameters(currentParam, _K, _omAll, _tAll, _D, _xi);

    //double repr = internal::computeMeanReproErr(_patternPoints, _imagePoints, _K, _D, _xi, _omAll, _tAll);

    if (omAll.needed())
    {
        omAll.create((int)_omAll.size(), 1, CV_64FC3);
    }
    if (tAll.needed())
    {
        tAll.create((int)_tAll.size(), 1, CV_64FC3);
    }
    if (omAll.kind() == _InputArray::STD_VECTOR_MAT)
    {
        for (int i = 0; i < n; ++i)
        {
            omAll.create(3, 1, CV_64F, i, true);
            tAll.create(3, 1, CV_64F, i, true);
            Mat tmpom = Mat(_omAll[i]);
            Mat tmpt = Mat(_tAll[i]);
            tmpom.convertTo(tmpom, CV_64F);
            tmpt.convertTo(tmpt, CV_64F);
            tmpom.copyTo(omAll.getMat(i));
            tmpt.copyTo(tAll.getMat(i));
        }
    }
    else
    {
        Mat(_omAll).convertTo(omAll, CV_64FC3);
        Mat(_tAll).convertTo(tAll, CV_64FC3);
    }

    if(K.empty())
    {
        K.create(3, 3, CV_64F);
    }
    if (D.empty())
    {
        D.create(1, 4, CV_64F);
    }

    Mat(_K).convertTo(K.getMat(), K.empty()? CV_64F : K.type());
    Mat(_D).convertTo(D.getMat(), D.empty() ? CV_64F: D.type());

    if (xi.empty())
    {
        xi.create(1, 1, CV_64F);
    }
    Mat xi_m = Mat(1, 1, CV_64F);
    xi_m.at<double>(0) = _xi;
    xi_m.convertTo(xi.getMat(), xi.empty() ? CV_64F : xi.type());

    if (idx.needed())
    {
        idx.create(1, (int)_idx.total(), CV_32S);
        _idx.copyTo(idx.getMat());
    }

    Vec2d std_error;
    double rms;
    Mat errors;
    cv::omnidir::internal::estimateUncertainties(_patternPoints, _imagePoints, finalParam, errors, std_error, rms, flags);
    return rms;
}

double cv::omnidir::stereoCalibrate(InputOutputArrayOfArrays objectPoints, InputOutputArrayOfArrays imagePoints1, InputOutputArrayOfArrays imagePoints2,
    const Size& imageSize1, const Size& imageSize2, InputOutputArray K1, InputOutputArray xi1, InputOutputArray D1, InputOutputArray K2, InputOutputArray xi2,
    InputOutputArray D2, OutputArray om, OutputArray T, OutputArrayOfArrays omL, OutputArrayOfArrays tL, int flags, TermCriteria criteria, OutputArray idx)
{
    CV_Assert(!objectPoints.empty() && (objectPoints.type() == CV_64FC3 || objectPoints.type() == CV_32FC3));
    CV_Assert(!imagePoints1.empty() && (imagePoints1.type() == CV_64FC2 || imagePoints1.type() == CV_32FC2));
    CV_Assert(!imagePoints2.empty() && (imagePoints2.type() == CV_64FC2 || imagePoints2.type() == CV_32FC2));

    CV_Assert(((flags & CALIB_USE_GUESS) && !K1.empty() && !D1.empty() && !K2.empty() && !D2.empty()) || !(flags & CALIB_USE_GUESS));

    int depth = objectPoints.depth();

    std::vector<Mat> _objectPoints, _imagePoints1, _imagePoints2,
					_objectPointsFilt, _imagePoints1Filt, _imagePoints2Filt;
    for (int i = 0; i < (int)objectPoints.total(); ++i)
    {
        _objectPoints.push_back(objectPoints.getMat(i));
        _imagePoints1.push_back(imagePoints1.getMat(i));
        _imagePoints2.push_back(imagePoints2.getMat(i));
        if (depth == CV_32F)
        {
            _objectPoints[i].convertTo(_objectPoints[i], CV_64FC3);
            _imagePoints1[i].convertTo(_imagePoints1[i], CV_64FC2);
            _imagePoints2[i].convertTo(_imagePoints2[i], CV_64FC2);
        }
    }

    Matx33d _K1, _K2;
    Matx14d _D1, _D2;

    double _xi1, _xi2;

    std::vector<Vec3d> _omL, _TL;
    Vec3d _om, _T;

    // initializaition
    Mat _idx;
    internal::initializeStereoCalibration(_objectPoints, _imagePoints1, _imagePoints2, imageSize1, imageSize2, _om, _T, _omL, _TL, _K1, _D1, _K2, _D2, _xi1, _xi2, flags, _idx);
    if(idx.needed())
    {
        idx.create(1, (int)_idx.total(), CV_32S);
        _idx.copyTo(idx.getMat());
    }
	for (int i = 0; i < (int)_idx.total(); ++i)
	{
		_objectPointsFilt.push_back(_objectPoints[_idx.at<int>(i)]);
		_imagePoints1Filt.push_back(_imagePoints1[_idx.at<int>(i)]);
		_imagePoints2Filt.push_back(_imagePoints2[_idx.at<int>(i)]);
	}

    int n = (int)_objectPointsFilt.size();
    Mat finalParam(1, 10 + 6*n, CV_64F);
    Mat currentParam(1, 10 + 6*n, CV_64F);

    //double repr1 = internal::computeMeanReproErrStereo(_objectPoints, _imagePoints1, _imagePoints2, _K1, _K2, _D1, _D2, _xi1, _xi2, _om,
    //    _T, _omL, _TL);
    cv::omnidir::internal::encodeParametersStereo(_K1, _K2, _om, _T, _omL, _TL, _D1, _D2, _xi1, _xi2, currentParam);

    // optimization
    const double alpha_smooth = 0.01;
    double change = 1;
    for(int iter = 0; ; ++iter)
    {
        if ((criteria.type == 1 && iter >= criteria.maxCount)  ||
            (criteria.type == 2 && change <= criteria.epsilon) ||
            (criteria.type == 3 && (change <= criteria.epsilon || iter >= criteria.maxCount)))
            break;
        double alpha_smooth2 = 1 - std::pow(1 - alpha_smooth, (double)iter + 1.0);
        Mat JTJ_inv, JTError;
		double epsilon = 0.01 * std::pow(0.9, (double)iter/10);

        cv::omnidir::internal::computeJacobianStereo(_objectPointsFilt, _imagePoints1Filt, _imagePoints2Filt, currentParam,
			JTJ_inv, JTError, flags, epsilon);

        // Gauss¨CNewton
        Mat G = alpha_smooth2*JTJ_inv * JTError;

        omnidir::internal::fillFixedStereo(G, flags, n);

        finalParam = currentParam + G.t();

        change = norm(G) / norm(currentParam);

        currentParam = finalParam.clone();
        cv::omnidir::internal::decodeParametersStereo(currentParam, _K1, _K2, _om, _T, _omL, _TL, _D1, _D2, _xi1, _xi2);
        //double repr = internal::computeMeanReproErrStereo(_objectPoints, _imagePoints1, _imagePoints2, _K1, _K2, _D1, _D2, _xi1, _xi2, _om,
        //    _T, _omL, _TL);

    }
    cv::omnidir::internal::decodeParametersStereo(finalParam, _K1, _K2, _om, _T, _omL, _TL, _D1, _D2, _xi1, _xi2);
    //double repr = internal::computeMeanReproErrStereo(_objectPoints, _imagePoints1, _imagePoints2, _K1, _K2, _D1, _D2, _xi1, _xi2, _om,
    //    _T, _omL, _TL);

    if (K1.empty())
    {
        K1.create(3, 3, CV_64F);
        D1.create(1, 4, CV_64F);
        K2.create(3, 3, CV_64F);
        D2.create(1, 4, CV_64F);
    }
    if (om.empty())
    {
        om.create(1, 3, CV_64F);
        T.create(1, 3, CV_64F);
    }
    if (omL.empty())
    {
        omL.create(1, n, CV_64FC3);
        tL.create(1, n, CV_64FC3);
    }

    Mat(_K1).convertTo(K1.getMat(), K1.empty() ? CV_64F : K1.type());
    Mat(_D1).convertTo(D1.getMat(), D1.empty() ? CV_64F : D1.type());
    Mat(_K2).convertTo(K2.getMat(), K2.empty() ? CV_64F : K2.type());
    Mat(_D2).convertTo(D2.getMat(), D2.empty() ? CV_64F : D2.type());

    Mat(_om).convertTo(om.getMat(), om.empty() ? CV_64F: om.type());
    Mat(_T).convertTo(T.getMat(), T.empty() ? CV_64F: T.type());

    if (omL.needed())
    {
        omL.create((int)_omL.size(), 1, CV_64FC3);
    }
    if (tL.needed())
    {
        tL.create((int)_TL.size(), 1, CV_64FC3);
    }

    if (omL.kind() == _InputArray::STD_VECTOR_MAT)
    {
        for (int i = 0; i < n; ++i)
        {
            omL.create(3, 1, CV_64F, i, true);
            tL.create(3, 1, CV_64F, i, true);
            Mat(_omL[i]).copyTo(omL.getMat(i));
            Mat(_TL[i]).copyTo(tL.getMat(i));
        }
    }
    else
    {
        Mat(_omL).convertTo(omL, omL.empty() ? CV_64FC3 : omL.type());
        Mat(_TL).convertTo(tL, tL.empty() ? CV_64FC3 : tL.type());
    }

    Mat xi1_m = Mat(1, 1, CV_64F),
        xi2_m = Mat(1, 1, CV_64F);
    xi1_m.at<double>(0) = _xi1;
    xi2_m.at<double>(0) = _xi2;

    if (xi1.empty())
    {
        xi1.create(1, 1, CV_64F);
    }
    if (xi2.empty())
    {
        xi2.create(1, 1, CV_64F);
    }
    xi1_m.convertTo(xi1, xi1.empty() ? CV_64F : xi1.type());
    xi2_m.convertTo(xi2, xi2.empty() ? CV_64F : xi2.type());

    // compute uncertainty
    Vec2d std_error;
    double rms;
    Mat errors;

    cv::omnidir::internal::estimateUncertaintiesStereo(_objectPointsFilt, _imagePoints1Filt, _imagePoints2Filt,
		finalParam, errors, std_error, rms, flags);
    return rms;
}

void cv::omnidir::stereoReconstruct(InputArray image1, InputArray image2, InputArray K1, InputArray D1,
    InputArray xi1, InputArray K2, InputArray D2, InputArray xi2, InputArray R, InputArray T, int flag,
    int numDisparities, int SADWindowSize, OutputArray disparity, OutputArray image1Rec, OutputArray image2Rec,
    const Size& newSize, InputArray Knew, OutputArray pointCloud, int pointType)
{
    CV_Assert(!K1.empty() && K1.size() == Size(3,3) && (K1.type() == CV_64F || K1.type() == CV_32F));
    CV_Assert(!K2.empty() && K2.size() == Size(3,3) && (K2.type() == CV_64F || K2.type() == CV_32F));
    CV_Assert(!D1.empty() && D1.total() == 4 && (D1.type() == CV_64F || D1.type() == CV_32F));
    CV_Assert(!D2.empty() && D2.total() == 4 && (D2.type() == CV_64F || D2.type() == CV_32F));
    CV_Assert(!R.empty() && (R.size() == Size(3,3) || R.total() == 3) && (R.type() == CV_64F || R.type() == CV_32F));
    CV_Assert(!T.empty() && T.total() == 3 && (T.type() == CV_64F || T.type() == CV_32F));
    CV_Assert(!image1.empty() && (image1.type() == CV_8U || image1.type() == CV_8UC3));
    CV_Assert(!image2.empty() && (image2.type() == CV_8U || image2.type() == CV_8UC3));
    CV_Assert(flag == omnidir::RECTIFY_LONGLATI || flag == omnidir::RECTIFY_PERSPECTIVE);

    Mat _K1, _D1, _K2, _D2, _R, _T;

    K1.getMat().convertTo(_K1, CV_64F);
    K2.getMat().convertTo(_K2, CV_64F);
    D1.getMat().convertTo(_D1, CV_64F);
    D2.getMat().convertTo(_D2, CV_64F);
    T.getMat().reshape(1, 3).convertTo(_T, CV_64F);

    if (R.size() == Size(3, 3))
    {
        R.getMat().convertTo(_R, CV_64F);
    }
    else if (R.total() == 3)
    {
        Rodrigues(R.getMat(), _R);
        _R.convertTo(_R, CV_64F);
    }
    // stereo rectify so that stereo matching can be applied in one line
    Mat R1, R2;
    stereoRectify(_R, _T, R1, R2);
    Mat undis1, undis2;
    Matx33d _Knew = Matx33d(_K1);
    if (!Knew.empty())
    {
        Knew.getMat().convertTo(_Knew, CV_64F);
    }

    undistortImage(image1.getMat(), undis1, _K1, _D1, xi1, flag, _Knew, newSize, R1);
    undistortImage(image2.getMat(), undis2, _K2, _D2, xi2, flag, _Knew, newSize, R2);

    undis1.copyTo(image1Rec);
    undis2.copyTo(image2Rec);

    // stereo matching by semi-global
    Mat _disMap;
    int channel = image1.channels();

    //cv::StereoSGBM matching(0, numDisparities, SADWindowSize, 8*channel*SADWindowSize*SADWindowSize, 32*channel*SADWindowSize*SADWindowSize);
    //matching(undis1, undis2, _depthMap);
	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, numDisparities, SADWindowSize, 8 * channel*SADWindowSize*SADWindowSize, 32 * channel*SADWindowSize*SADWindowSize);

	sgbm->compute(undis1, undis2, _disMap);

    // some regions of image1 is black, the corresponding regions of disparity map is also invalid.
    Mat realDis;
	_disMap.convertTo(_disMap, CV_32F);
    Mat(_disMap/16.0f).convertTo(realDis, CV_32F);

    Mat grayImg, binaryImg, idx;
    if (undis1.channels() == 3)
    {
        cvtColor(undis1, grayImg, COLOR_RGB2GRAY);
    }
    else
    {
        grayImg = undis1;
    }

    binaryImg = (grayImg <= 0);
    findNonZero(binaryImg, idx);
    for (int i = 0; i < (int)idx.total(); ++i)
    {
        Vec2i _idx = idx.at<Vec2i>(i);
        realDis.at<float>(_idx[1], _idx[0]) = 0.0f;
    }

    disparity.create(realDis.size(), realDis.type());
    realDis.copyTo(disparity.getMat());

    std::vector<Vec3f> _pointCloud;
    std::vector<Vec6f> _pointCloudColor;
    double baseline = cv::norm(T);
    double f = _Knew(0, 0);
    Matx33d K_inv = _Knew.inv();

    std::vector<Mat> rgb;
    if (undis1.channels() == 3)
    {
        split(undis1, rgb);
    }

    if (pointCloud.needed())
    {
        for (int i = 0; i < newSize.width; ++i)
        {
            for(int j = 0; j < newSize.height; ++j)
            {
                Vec3f point;
                Vec6f pointColor;
                if (realDis.at<float>(j, i) > 15)
                {
                    float depth = float(baseline * f /realDis.at<float>(j, i));
                    // for RECTIFY_PERSPECTIVE, (x,y) are image plane points,
                    // for RECTIFY_LONGLATI, (x,y) are (theta, phi) angles
                    float x = float(K_inv(0,0) * i + K_inv(0,1) * j + K_inv(0,2));
                    float y = float(K_inv(1,0) * i + K_inv(1,1) * j + K_inv(1,2));
                    if (flag == omnidir::RECTIFY_LONGLATI)
                    {
                        point = Vec3f((float)-std::cos(x), (float)(-std::sin(x)*std::cos(y)), (float)(std::sin(x)*std::sin(y))) * depth;
                    }
                    else if(flag == omnidir::RECTIFY_PERSPECTIVE)
                    {
                        point = Vec3f(float(x), float(y), 1.0f) * depth;
                    }
                    if (pointType == XYZ)
                    {
                        _pointCloud.push_back(point);
                    }
                    else if (pointType == XYZRGB)
                    {
                        pointColor[0] = point[0];
                        pointColor[1] = point[1];
                        pointColor[2] = point[2];

                        if (undis1.channels() == 1)
                        {
                            pointColor[3] = float(undis1.at<uchar>(j, i));
                            pointColor[4] = pointColor[3];
                            pointColor[5] = pointColor[3];
                        }
                        else if (undis1.channels() == 3)
                        {
                            pointColor[3] = rgb[0].at<uchar>(j, i);
                            pointColor[4] = rgb[1].at<uchar>(j, i);
                            pointColor[5] = rgb[2].at<uchar>(j, i);
                        }
                        _pointCloudColor.push_back(pointColor);
                    }
                }
            }
        }

        if (pointType == XYZ)
        {
            Mat(_pointCloud).convertTo(pointCloud, CV_MAKE_TYPE(CV_32F, 3));
        }
        else if (pointType == XYZRGB)
        {
            Mat(_pointCloudColor).convertTo(pointCloud, CV_MAKE_TYPE(CV_32F, 6));
        }
    }
}

void cv::omnidir::internal::encodeParameters(InputArray K, InputArrayOfArrays omAll, InputArrayOfArrays tAll, InputArray distoaration, double xi, OutputArray parameters)
{
    CV_Assert(K.type() == CV_64F && K.size() == Size(3,3));
    CV_Assert(distoaration.total() == 4 && distoaration.type() == CV_64F);
    int n = (int)omAll.total();
    Mat _omAll = omAll.getMat(), _tAll = tAll.getMat();

    Matx33d _K = K.getMat();
    Vec4d _D = (Vec4d)distoaration.getMat();
    parameters.create(1, 10+6*n,CV_64F);
    Mat _params = parameters.getMat();
    for (int i = 0; i < n; i++)
    {
        Mat(_omAll.at<Vec3d>(i)).reshape(1, 1).copyTo(_params.colRange(i*6, i*6+3));
        Mat(_tAll.at<Vec3d>(i)).reshape(1, 1).copyTo(_params.colRange(i*6+3, (i+1)*6));
    }

    _params.at<double>(0, 6*n) = _K(0,0);
    _params.at<double>(0, 6*n+1) = _K(1,1);
    _params.at<double>(0, 6*n+2) = _K(0,1);
    _params.at<double>(0, 6*n+3) = _K(0,2);
    _params.at<double>(0, 6*n+4) = _K(1,2);
    _params.at<double>(0, 6*n+5) = xi;
    _params.at<double>(0, 6*n+6) = _D[0];
    _params.at<double>(0, 6*n+7) = _D[1];
    _params.at<double>(0, 6*n+8) = _D[2];
    _params.at<double>(0, 6*n+9) = _D[3];
}

void cv::omnidir::internal::encodeParametersStereo(InputArray K1, InputArray K2, InputArray om, InputArray T, InputArrayOfArrays omL, InputArrayOfArrays tL,
    InputArray D1, InputArray D2, double xi1, double xi2, OutputArray parameters)
{
    CV_Assert(!K1.empty() && K1.type() == CV_64F && K1.size() == Size(3,3));
    CV_Assert(!K2.empty() && K2.type() == CV_64F && K2.size() == Size(3,3));
    CV_Assert(!om.empty() && om.type() == CV_64F && om.total() == 3);
    CV_Assert(!T.empty() && T.type() == CV_64F && T.total() == 3);
    CV_Assert(omL.total() == tL.total() && omL.type() == CV_64FC3 && tL.type() == CV_64FC3);
    CV_Assert(D1.type() == CV_64F && D1.total() == 4 && D2.type() == CV_64F && D2.total() == 4);

    int n = (int)omL.total();
    // om, T, omL, tL, intrinsic left, intrinsic right
    parameters.create(1, 20 + 6 * (n + 1), CV_64F);

    Mat _params = parameters.getMat();

    om.getMat().reshape(1, 1).copyTo(_params.colRange(0, 3));
    T.getMat().reshape(1, 1).copyTo(_params.colRange(3, 6));
    for(int i = 0; i < n; ++i)
    {
        Mat(omL.getMat().at<Vec3d>(i)).reshape(1, 1).copyTo(_params.colRange(6 + i*6, 6 + i*6 + 3));
        Mat(tL.getMat().at<Vec3d>(i)).reshape(1, 1).copyTo(_params.colRange(6 + i*6 + 3, 6 + i*6 + 6));
    }

    Matx33d _K1 = K1.getMat();
    Matx33d _K2 = K2.getMat();
    Vec4d _D1 = D1.getMat();
    Vec4d _D2 = D2.getMat();
    _params.at<double>(0, 6*(n+1)) = _K1(0,0);
    _params.at<double>(0, 6*(n+1)+1) = _K1(1,1);
    _params.at<double>(0, 6*(n+1)+2) = _K1(0,1);
    _params.at<double>(0, 6*(n+1)+3) = _K1(0,2);
    _params.at<double>(0, 6*(n+1)+4) = _K1(1,2);
    _params.at<double>(0, 6*(n+1)+5) = xi1;
    _params.at<double>(0, 6*(n+1)+6) = _D1[0];
    _params.at<double>(0, 6*(n+1)+7) = _D1[1];
    _params.at<double>(0, 6*(n+1)+8) = _D1[2];
    _params.at<double>(0, 6*(n+1)+9) = _D1[3];

    _params.at<double>(0, 6*(n+1)+10) = _K2(0,0);
    _params.at<double>(0, 6*(n+1)+11) = _K2(1,1);
    _params.at<double>(0, 6*(n+1)+12) = _K2(0,1);
    _params.at<double>(0, 6*(n+1)+13) = _K2(0,2);
    _params.at<double>(0, 6*(n+1)+14) = _K2(1,2);
    _params.at<double>(0, 6*(n+1)+15) = xi2;
    _params.at<double>(0, 6*(n+1)+16) = _D2[0];
    _params.at<double>(0, 6*(n+1)+17) = _D2[1];
    _params.at<double>(0, 6*(n+1)+18) = _D2[2];
    _params.at<double>(0, 6*(n+1)+19) = _D2[3];
}


 void cv::omnidir::internal::decodeParameters(InputArray parameters, OutputArray K, OutputArrayOfArrays omAll, OutputArrayOfArrays tAll, OutputArray distoration, double& xi)
 {
    if(K.empty())
        K.create(3,3,CV_64F);
    Matx33d _K;
    int n = (int)(parameters.total()-10)/6;
    if(omAll.empty())
        omAll.create(1, n, CV_64FC3);
    if(tAll.empty())
        tAll.create(1, n, CV_64FC3);
    if(distoration.empty())
        distoration.create(1, 4, CV_64F);
    Matx14d _D = distoration.getMat();
    Mat param = parameters.getMat();
    double *para = param.ptr<double>();
    _K = Matx33d(para[6*n], para[6*n+2], para[6*n+3],
        0,    para[6*n+1], para[6*n+4],
        0,    0,  1);
    _D  = Matx14d(para[6*n+6], para[6*n+7], para[6*n+8], para[6*n+9]);
    xi = para[6*n+5];
    std::vector<Vec3d> _omAll(n), _tAll(n);
    for (int i = 0; i < n; i++)
    {
        _omAll[i] = Vec3d(param.colRange(i*6, i*6+3));
        _tAll[i] = Vec3d(param.colRange(i*6+3, i*6+6));
    }
    Mat(_D).convertTo(distoration, CV_64F);
    Mat(_K).convertTo(K, CV_64F);

    if (omAll.kind() == _InputArray::STD_VECTOR_MAT)
    {
        for (int i = 0; i < n; ++i)
        {
            Mat(_omAll[i]).copyTo(omAll.getMat(i));
            Mat(_tAll[i]).copyTo(tAll.getMat(i));
        }
    }
    else
    {
        Mat(_omAll).convertTo(omAll, CV_64FC3);
        Mat(_tAll).convertTo(tAll, CV_64FC3);
    }
}

 void cv::omnidir::internal::decodeParametersStereo(InputArray parameters, OutputArray K1, OutputArray K2, OutputArray om, OutputArray T, OutputArrayOfArrays omL,
     OutputArrayOfArrays tL, OutputArray D1, OutputArray D2, double& xi1, double& xi2)
 {
    if(K1.empty())
        K1.create(3, 3, CV_64F);
    if(K2.empty())
        K2.create(3, 3, CV_64F);
    if(om.empty())
        om.create(3, 1, CV_64F);
    if(T.empty())
        T.create(3, 1, CV_64F);

    int n = ((int)parameters.total() - 20) / 6 - 1;

    if(omL.empty())
        omL.create(1, n, CV_64FC3);
    if(tL.empty())
        tL.create(1, n, CV_64FC3);
    if(D1.empty())
        D1.create(1, 4, CV_64F);
    if(D2.empty())
        D2.create(1, 4, CV_64F);

    Mat param = parameters.getMat().reshape(1, 1);
    param.colRange(0, 3).reshape(1, 3).copyTo(om.getMat());
    param.colRange(3, 6).reshape(1, 3).copyTo(T.getMat());
    std::vector<Vec3d> _omL, _tL;

    for(int i = 0; i < n; i++)
    {
        _omL.push_back(Vec3d(param.colRange(6 + i*6, 6 + i*6 + 3)));
        _tL.push_back(Vec3d(param.colRange(6 + i*6 + 3, 6 + i*6 + 6)));
    }

    double* para = param.ptr<double>();
    int offset1 = (n + 1)*6;
    Matx33d _K1(para[offset1], para[offset1+2], para[offset1+3],
                0,      para[offset1+1],     para[offset1+4],
                0,          0,                  1);
    xi1 = para[offset1+5];
    Matx14d _D1(para[offset1+6], para[offset1+7], para[offset1+8], para[offset1+9]);

    int offset2 = (n + 1)*6 + 10;
    Matx33d _K2(para[offset2], para[offset2+2], para[offset2+3],
                0,      para[offset2+1],     para[offset2+4],
                0,          0,                  1);
    xi2 = para[offset2+5];
    Matx14d _D2(para[offset2+6], para[offset2+7], para[offset2+8], para[offset2+9]);

    Mat(_K1).convertTo(K1, CV_64F);
    Mat(_D1).convertTo(D1, CV_64F);
    Mat(_K2).convertTo(K2, CV_64F);
    Mat(_D2).convertTo(D2, CV_64F);
    if(omL.kind() == _InputArray::STD_VECTOR_MAT)
    {
        for(int i = 0; i < n; ++i)
        {
            Mat(_omL[i]).copyTo(omL.getMat(i));
            Mat(_tL[i]).copyTo(tL.getMat(i));
        }
    }
    else
    {
        Mat(_omL).convertTo(omL, CV_64FC3);
        Mat(_tL).convertTo(tL, CV_64FC3);
    }
 }

void cv::omnidir::internal::estimateUncertainties(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, InputArray parameters,
    Mat& errors, Vec2d& std_error, double& rms, int flags)
{
    CV_Assert(!objectPoints.empty() && objectPoints.type() == CV_64FC3);
    CV_Assert(!imagePoints.empty() && imagePoints.type() == CV_64FC2);
    CV_Assert(!parameters.empty() && parameters.type() == CV_64F);

    int n = (int) objectPoints.total();
    // assume every image has the same number of objectpoints
    int nPointsAll = 0;
    for (int i = 0; i < n; ++i)
    {
        nPointsAll += (int)objectPoints.getMat(i).total();
    }

    Mat reprojError = Mat(nPointsAll, 1, CV_64FC2);

    double* para = parameters.getMat().ptr<double>();
    Matx33d K(para[6*n], para[6*n+2], para[6*n+3],
              0,    para[6*n+1], para[6*n+4],
              0,    0,  1);
    Matx14d D(para[6*n+6], para[6*n+7], para[6*n+8], para[6*n+9]);
    double xi = para[6*n+5];
    int nPointsAccu = 0;

    for(int i=0; i < n; ++i)
    {
		Mat imgPoints, objPoints;
		imagePoints.getMat(i).copyTo(imgPoints);
		objectPoints.getMat(i).copyTo(objPoints);
		imgPoints = imgPoints.reshape(2, imgPoints.rows*imgPoints.cols);
		objPoints = objPoints.reshape(3, objPoints.rows*objPoints.cols);

        Mat om = parameters.getMat().colRange(i*6, i*6+3);
        Mat T = parameters.getMat().colRange(i*6+3, (i+1)*6);

        Mat x;
        omnidir::projectPoints(objPoints, x, om, T, K, xi, D, cv::noArray());

        Mat errorx = (imgPoints - x);

        //reprojError.rowRange(errorx.rows*i, errorx.rows*(i+1)) = errorx.clone();
        errorx.copyTo(reprojError.rowRange(nPointsAccu, nPointsAccu + (int)errorx.total()));
        nPointsAccu += (int)errorx.total();
    }

    meanStdDev(reprojError, noArray(), std_error);
    std_error *= sqrt((double)reprojError.total()/((double)reprojError.total() - 1.0));

    Mat sigma_x;
    meanStdDev(reprojError.reshape(1,1), noArray(), sigma_x);
    sigma_x *= sqrt(2.0*(double)reprojError.total()/(2.0*(double)reprojError.total() - 1.0));
    double s = sigma_x.at<double>(0);

    Mat _JTJ_inv, _JTE;
    computeJacobian(objectPoints, imagePoints, parameters, _JTJ_inv, _JTE, flags, 0.0);
    sqrt(_JTJ_inv, _JTJ_inv);

    errors = 3 * s * _JTJ_inv.diag();

    checkFixed(errors, flags, n);

    rms = 0;
    const Vec2d* ptr_ex = reprojError.ptr<Vec2d>();
    for (int i = 0; i < (int)reprojError.total(); i++)
    {
        rms += ptr_ex[i][0] * ptr_ex[i][0] + ptr_ex[i][1] * ptr_ex[i][1];
    }

    rms /= (double)reprojError.total();
    rms = sqrt(rms);
}

// estimateUncertaintiesStereo
void cv::omnidir::internal::estimateUncertaintiesStereo(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
    InputArray parameters, Mat& errors, Vec2d& std_error, double& rms, int flags)
{
    CV_Assert(!objectPoints.empty() && objectPoints.type() == CV_64FC3);
    CV_Assert(!imagePoints1.empty() && imagePoints1.type() == CV_64FC2 && imagePoints1.total() == objectPoints.total());
    CV_Assert(!imagePoints2.empty() && imagePoints2.type() == CV_64FC2 && imagePoints1.total() == imagePoints2.total());
    int n_img = (int)objectPoints.total();
    CV_Assert((int)parameters.total() == (6*(n_img+1)+20));

    Mat _K1, _K2, _D1, _D2;
    Vec3d _om, _T;
    std::vector<Vec3d> _omL(n_img), _tL(n_img);
    Mat _parameters = parameters.getMat().reshape(1, 1);
    double _xi1, _xi2;
    internal::decodeParametersStereo(_parameters, _K1, _K2, _om, _T, _omL, _tL, _D1, _D2, _xi1, _xi2);

    int n_points = (int)objectPoints.getMat(0).total();
    Mat reprojErrorAll = Mat::zeros(2*n_points*n_img, 1, CV_64FC2);

    // error for left image
    for (int i = 0; i < n_img; ++i)
    {
        Mat objPointsi, imgPointsi;
        objectPoints.getMat(i).copyTo(objPointsi);
        imagePoints1.getMat(i).copyTo(imgPointsi);
        objPointsi = objPointsi.reshape(3, objPointsi.rows*objPointsi.cols);
        imgPointsi = imgPointsi.reshape(2, imgPointsi.rows*imgPointsi.cols);

        Mat x;
        omnidir::projectPoints(objPointsi, x, _omL[i], _tL[i], _K1, _xi1, _D1, cv::noArray());

        Mat errorx = imgPointsi - x;

        errorx.copyTo(reprojErrorAll.rowRange(i*2*n_points, (i*2+1)*n_points));
    }
    // error for right image
    for (int i = 0; i < n_img; ++i)
    {
        Mat objPointsi, imgPointsi;
        objectPoints.getMat(i).copyTo(objPointsi);
        imagePoints2.getMat(i).copyTo(imgPointsi);
        objPointsi = objPointsi.reshape(3, objPointsi.rows*objPointsi.cols);
        imgPointsi = imgPointsi.reshape(2, imgPointsi.rows*imgPointsi.cols);

        Mat x;
        Mat _R, _R2, _R1, _T2, _T1, _om2;
        Rodrigues(_om, _R);
        Rodrigues(_omL[i], _R1);
        _T1 = Mat(_tL[i]);
        _R2 = _R * _R1;
        _T2 = _R * _T1 + Mat(_T);
        Rodrigues(_R2, _om2);

        omnidir::projectPoints(objPointsi, x, _om2, _T2, _K2, _xi2, _D2, cv::noArray());

        Mat errorx = imgPointsi - x;

        errorx.copyTo(reprojErrorAll.rowRange((i*2+1)*n_points, (i+1)*2*n_points));
    }

    meanStdDev(reprojErrorAll, cv::noArray(), std_error);
    std_error *= sqrt((double)reprojErrorAll.total()/((double)reprojErrorAll.total() - 1.0));

    Mat sigma_x;
    meanStdDev(reprojErrorAll.reshape(1,1), noArray(), sigma_x);
    sigma_x *= sqrt(2.0*(double)reprojErrorAll.total()/(2.0*(double)reprojErrorAll.total() - 1.0));
    double s = sigma_x.at<double>(0);

    Mat _JTJ_inv, _JTE;
    computeJacobianStereo(objectPoints, imagePoints1, imagePoints2, _parameters, _JTJ_inv, _JTE, flags, 0.0);
    cv::sqrt(_JTJ_inv, _JTJ_inv);

    errors = 3 * s * _JTJ_inv.diag();

    rms = 0;

    const Vec2d* ptr_ex = reprojErrorAll.ptr<Vec2d>();
    for (int i = 0; i < (int)reprojErrorAll.total(); i++)
    {
        rms += ptr_ex[i][0] * ptr_ex[i][0] + ptr_ex[i][1] * ptr_ex[i][1];
    }
    rms /= (double)reprojErrorAll.total();
    rms = sqrt(rms);
}

//
double cv::omnidir::internal::computeMeanReproErr(InputArrayOfArrays imagePoints, InputArrayOfArrays proImagePoints)
{
    CV_Assert(!imagePoints.empty() && imagePoints.type()==CV_64FC2);
    CV_Assert(!proImagePoints.empty() && proImagePoints.type() == CV_64FC2);
    CV_Assert(imagePoints.total() == proImagePoints.total());

    int n = (int)imagePoints.total();
    double reprojError = 0;
    int totalPoints = 0;
    if (imagePoints.kind() == _InputArray::STD_VECTOR_MAT)
    {
        for (int i = 0; i < n; i++)
        {
			Mat x, proj_x;
			imagePoints.getMat(i).copyTo(x);
			proImagePoints.getMat(i).copyTo(proj_x);
			Mat errorI = x.reshape(2, x.rows*x.cols) - proj_x.reshape(2, proj_x.rows*proj_x.cols);
            //Mat errorI = imagePoints.getMat(i) - proImagePoints.getMat(i);
            totalPoints += (int)errorI.total();
            Vec2d* ptr_err = errorI.ptr<Vec2d>();
            for (int j = 0; j < (int)errorI.total(); j++)
            {
                reprojError += sqrt(ptr_err[j][0]*ptr_err[j][0] + ptr_err[j][1]*ptr_err[j][1]);
            }
        }
    }
    else
    {
		Mat x, proj_x;
		imagePoints.getMat().copyTo(x);
		proImagePoints.getMat().copyTo(proj_x);
		Mat errorI = x.reshape(2, x.rows*x.cols) - proj_x.reshape(2, proj_x.rows*proj_x.cols);
        //Mat errorI = imagePoints.getMat() - proImagePoints.getMat();
        totalPoints += (int)errorI.total();
        Vec2d* ptr_err = errorI.ptr<Vec2d>();
        for (int j = 0; j < (int)errorI.total(); j++)
        {
            reprojError += sqrt(ptr_err[j][0]*ptr_err[j][0] + ptr_err[j][1]*ptr_err[j][1]);
        }
    }
    double meanReprojError = reprojError / totalPoints;
    return meanReprojError;
}

double cv::omnidir::internal::computeMeanReproErr(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, InputArray K, InputArray D, double xi, InputArrayOfArrays omAll,
    InputArrayOfArrays tAll)
{
    CV_Assert(objectPoints.total() == imagePoints.total());
    CV_Assert(!objectPoints.empty() && objectPoints.type() == CV_64FC3);
    CV_Assert(!imagePoints.empty() && imagePoints.type() == CV_64FC2);
    std::vector<Mat> proImagePoints;
    int n = (int)objectPoints.total();
    Mat _omAll = omAll.getMat();
    Mat _tAll = tAll.getMat();
    for(int i = 0; i < n; ++i)
    {
        Mat imgPoint;
        //cv::omnidir::projectPoints(objetPoints.getMat(i), imgPoint, omAll.getMat(i), tAll.getMat(i), K.getMat(), xi, D.getMat(), noArray());
        cv::omnidir::projectPoints(objectPoints.getMat(i), imgPoint, _omAll.at<Vec3d>(i), _tAll.at<Vec3d>(i), K.getMat(), xi, D.getMat(), noArray());
        proImagePoints.push_back(imgPoint);
    }

    return internal::computeMeanReproErr(imagePoints, proImagePoints);
}

double cv::omnidir::internal::computeMeanReproErrStereo(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2, InputArray K1, InputArray K2,
    InputArray D1, InputArray D2, double xi1, double xi2, InputArray om, InputArray T, InputArrayOfArrays omL, InputArrayOfArrays TL)
{
    CV_Assert(objectPoints.total() == imagePoints1.total() && imagePoints1.total() == imagePoints2.total());
    CV_Assert(!objectPoints.empty() && objectPoints.type() == CV_64FC3);
    CV_Assert(!imagePoints1.empty() && imagePoints1.type() == CV_64FC2);
    CV_Assert(!imagePoints2.empty() && imagePoints2.type() == CV_64FC2);

    std::vector<Mat> proImagePoints1, proImagePoints2;
    int n = (int)objectPoints.total();
    Mat _omL = omL.getMat(), _TL = TL.getMat();
    Mat _om = om.getMat(), _R, _T = T.getMat();
    Rodrigues(_om, _R);
    Mat _K1 = K1.getMat(), _K2 = K2.getMat();
    Mat _D1 = D1.getMat(), _D2 = D2.getMat();

    // reprojection error for left image
    for (int i = 0; i < n; ++i)
    {
        Mat imgPoints;
        cv::omnidir::projectPoints(objectPoints.getMat(i), imgPoints, _omL.at<Vec3d>(i), _TL.at<Vec3d>(i), _K1, xi1, _D1, cv::noArray());
        proImagePoints1.push_back(imgPoints);
    }

    // reprojection error for right image
    for (int i = 0; i < n; ++i)
    {
        Mat imgPoints;
        Mat _omRi,_RRi,_TRi,_RLi, _TLi;
        Rodrigues(_omL.at<Vec3d>(i), _RLi);
        _TLi = Mat(_TL.at<Vec3d>(i)).reshape(1, 3);
        _RRi = _R * _RLi;
        _TRi = _R * _TLi + _T;
        Rodrigues(_RRi, _omRi);
        cv::omnidir::projectPoints(objectPoints.getMat(i), imgPoints, _omRi, _TRi, _K2, xi2, _D2, cv::noArray());
        proImagePoints2.push_back(imgPoints);
    }
    double reProErr1 = internal::computeMeanReproErr(imagePoints1, proImagePoints1);
    double reProErr2 = internal::computeMeanReproErr(imagePoints2, proImagePoints2);

    double reProErr = (reProErr1 + reProErr2) / 2.0;
    //double reProErr = reProErr1*reProErr1 + reProErr2* reProErr2;
    return reProErr;
}

void cv::omnidir::internal::checkFixed(Mat& G, int flags, int n)
{
    int _flags = flags;
    if(_flags >= omnidir::CALIB_FIX_CENTER)
    {
        G.at<double>(6*n+3) = 0;
        G.at<double>(6*n+4) = 0;
        _flags -= omnidir::CALIB_FIX_CENTER;
    }
    if(_flags >= omnidir::CALIB_FIX_GAMMA)
    {
        G.at<double>(6*n) = 0;
        G.at<double>(6*n+1) = 0;
        _flags -= omnidir::CALIB_FIX_GAMMA;
    }
    if(_flags >= omnidir::CALIB_FIX_XI)
    {
        G.at<double>(6*n + 5) = 0;
        _flags -= omnidir::CALIB_FIX_XI;
    }
    if(_flags >= omnidir::CALIB_FIX_P2)
    {
        G.at<double>(6*n + 9) = 0;
        _flags -= omnidir::CALIB_FIX_P2;
    }
    if(_flags >= omnidir::CALIB_FIX_P1)
    {
        G.at<double>(6*n + 8) = 0;
        _flags -= omnidir::CALIB_FIX_P1;
    }
    if(_flags >= omnidir::CALIB_FIX_K2)
    {
        G.at<double>(6*n + 7) = 0;
        _flags -= omnidir::CALIB_FIX_K2;
    }
    if(_flags >= omnidir::CALIB_FIX_K1)
    {
        G.at<double>(6*n + 6) = 0;
        _flags -= omnidir::CALIB_FIX_K1;
    }
    if(_flags >= omnidir::CALIB_FIX_SKEW)
    {
        G.at<double>(6*n + 2) = 0;
    }
}

// This function is from fisheye.cpp
void cv::omnidir::internal::subMatrix(const Mat& src, Mat& dst, const std::vector<int>& cols, const std::vector<int>& rows)
{
    CV_Assert(src.type() == CV_64FC1);

    int nonzeros_cols = cv::countNonZero(cols);
    Mat tmp(src.rows, nonzeros_cols, CV_64FC1);

    for (int i = 0, j = 0; i < (int)cols.size(); i++)
    {
        if (cols[i])
        {
            src.col(i).copyTo(tmp.col(j++));
        }
    }

    int nonzeros_rows  = cv::countNonZero(rows);
    Mat tmp1(nonzeros_rows, nonzeros_cols, CV_64FC1);
    for (int i = 0, j = 0; i < (int)rows.size(); i++)
    {
        if (rows[i])
        {
            tmp.row(i).copyTo(tmp1.row(j++));
        }
    }

    dst = tmp1.clone();
}

void cv::omnidir::internal::flags2idx(int flags, std::vector<int>& idx, int n)
{
    idx = std::vector<int>(6*n+10,1);
    int _flags = flags;
    if(_flags >= omnidir::CALIB_FIX_CENTER)
    {
        idx[6*n+3] = 0;
        idx[6*n+4] = 0;
        _flags -= omnidir::CALIB_FIX_CENTER;
    }
    if(_flags >= omnidir::CALIB_FIX_GAMMA)
    {
        idx[6*n] = 0;
        idx[6*n+1] = 0;
        _flags -= omnidir::CALIB_FIX_GAMMA;
    }
    if(_flags >= omnidir::CALIB_FIX_XI)
    {
        idx[6*n + 5] = 0;
        _flags -= omnidir::CALIB_FIX_XI;
    }
    if(_flags >= omnidir::CALIB_FIX_P2)
    {
        idx[6*n + 9] = 0;
        _flags -= omnidir::CALIB_FIX_P2;
    }
    if(_flags >= omnidir::CALIB_FIX_P1)
    {
        idx[6*n + 8] = 0;
        _flags -= omnidir::CALIB_FIX_P1;
    }
    if(_flags >= omnidir::CALIB_FIX_K2)
    {
        idx[6*n + 7] = 0;
        _flags -= omnidir::CALIB_FIX_K2;
    }
    if(_flags >= omnidir::CALIB_FIX_K1)
    {
        idx[6*n + 6] = 0;
        _flags -= omnidir::CALIB_FIX_K1;
    }
    if(_flags >= omnidir::CALIB_FIX_SKEW)
    {
        idx[6*n + 2] = 0;
    }
}

void cv::omnidir::internal::flags2idxStereo(int flags, std::vector<int>& idx, int n)
{
    idx = std::vector<int>(6*(n+1)+20, 1);
    int _flags = flags;
    int offset1 = 6*(n+1);
    int offset2 = offset1 + 10;
    if(_flags >= omnidir::CALIB_FIX_CENTER)
    {
        idx[offset1+3] = 0;
        idx[offset1+4] = 0;
        idx[offset2+3] = 0;
        idx[offset2+4] = 0;
        _flags -= omnidir::CALIB_FIX_CENTER;
    }
    if(_flags >= omnidir::CALIB_FIX_GAMMA)
    {
        idx[offset1] = 0;
        idx[offset1+1] = 0;
        idx[offset2] = 0;
        idx[offset2+1] = 0;
        _flags -= omnidir::CALIB_FIX_GAMMA;
    }
    if(_flags >= omnidir::CALIB_FIX_XI)
    {
        idx[offset1 + 5] = 0;
        idx[offset2 + 5] = 0;
        _flags -= omnidir::CALIB_FIX_XI;
    }
    if(_flags >= omnidir::CALIB_FIX_P2)
    {
        idx[offset1 + 9] = 0;
        idx[offset2 + 9] = 0;
        _flags -= omnidir::CALIB_FIX_P2;
    }
    if(_flags >= omnidir::CALIB_FIX_P1)
    {
        idx[offset1 + 8] = 0;
        idx[offset2 + 8] = 0;
        _flags -= omnidir::CALIB_FIX_P1;
    }
    if(_flags >= omnidir::CALIB_FIX_K2)
    {
        idx[offset1 + 7] = 0;
        idx[offset2 + 7] = 0;
        _flags -= omnidir::CALIB_FIX_K2;
    }
    if(_flags >= omnidir::CALIB_FIX_K1)
    {
        idx[offset1 + 6] = 0;
        idx[offset2 + 6] = 0;
        _flags -= omnidir::CALIB_FIX_K1;
    }
    if(_flags >= omnidir::CALIB_FIX_SKEW)
    {
        idx[offset1 + 2] = 0;
        idx[offset2 + 2] = 0;
    }
}

// fill in zerso for fixed parameters
void cv::omnidir::internal::fillFixed(Mat&G, int flags, int n)
{
    Mat tmp = G.clone();
    std::vector<int> idx(6*n + 10, 1);
    flags2idx(flags, idx, n);
    G.release();
    G.create(6*n +10, 1, CV_64F);
    G = cv::Mat::zeros(6*n +10, 1, CV_64F);
    for (int i = 0,j=0; i < (int)idx.size(); i++)
    {
        if (idx[i])
        {
            G.at<double>(i) = tmp.at<double>(j++);
        }
    }
}

void cv::omnidir::internal::fillFixedStereo(Mat& G, int flags, int n)
{
    Mat tmp = G.clone();
    std::vector<int> idx(6*(n+1)+20, 1);
    flags2idxStereo(flags, idx, n);
    G.release();
    G.create(6 * (n+1) + 20, 1, CV_64F);
    G = cv::Mat::zeros(6* (n + 1) + 20, 1, CV_64F);
    for (int i = 0,j=0; i < (int)idx.size(); i++)
    {
        if (idx[i])
        {
            G.at<double>(i) = tmp.at<double>(j++);
        }
    }
}

double cv::omnidir::internal::findMedian(const Mat& row)
{
    CV_Assert(!row.empty() && row.rows == 1 && row.type() == CV_64F);
    Mat tmp = row.clone();
    cv::sort(tmp, tmp, 0);
    if((int)tmp.total()%2 == 0)
        return tmp.at<double>((int)tmp.total() / 2);
    else
        return 0.5 * (tmp.at<double>((int)tmp.total() / 2) + tmp.at<double>((int)tmp.total()/2 - 1));
}

cv::Vec3d cv::omnidir::internal::findMedian3(InputArray mat)
{
    CV_Assert(mat.depth() == CV_64F && mat.getMat().rows == 1);
    Mat M = Mat(mat.getMat().t()).reshape(1).t();
    return Vec3d(findMedian(M.row(0)), findMedian(M.row(1)), findMedian(M.row(2)));
}

void cv::omnidir::stereoRectify(InputArray R, InputArray T, OutputArray R1, OutputArray R2)
{
    CV_Assert((R.size() == Size(3,3) || R.total() == 3) && (R.depth() == CV_32F || R.depth() == CV_64F));
    CV_Assert(T.total() == 3  && (T.depth() == CV_32F || T.depth() == CV_64F));

    Mat _R, _T;
    if (R.size() == Size(3, 3))
    {
        R.getMat().convertTo(_R, CV_64F);
    }
    else if (R.total() == 3)
    {
        Rodrigues(R.getMat(), _R);
        _R.convertTo(_R, CV_64F);
    }

    T.getMat().reshape(1, 3).convertTo(_T, CV_64F);

    R1.create(3, 3, CV_64F);
    Mat _R1 = R1.getMat();
    R2.create(3, 3, CV_64F);
    Mat _R2 = R2.getMat();

    Mat R21 = _R.t();
    Mat T21 = -_R.t() * _T;

    Mat e1, e2, e3;
    e1 = T21.t() / norm(T21);
    e2 = Mat(Matx13d(T21.at<double>(1)*-1, T21.at<double>(0), 0.0));
    e2 = e2 / norm(e2);
    e3 = e1.cross(e2);
    e3 = e3 / norm(e3);
    e1.copyTo(_R1.row(0));
    e2.copyTo(_R1.row(1));
    e3.copyTo(_R1.row(2));
    _R2 = R21 * _R1;

}

void cv::omnidir::internal::getInterset(InputArray idx1, InputArray idx2, OutputArray inter1, OutputArray inter2,
    OutputArray inter_ori)
{
    Mat _idx1 = idx1.getMat();
    Mat _idx2 = idx2.getMat();

    int n1 = (int)idx1.total();
    int n2 = (int)idx2.total();

    std::vector<int> _inter1, _inter2, _inter_ori;
    for (int i = 0; i < n1; ++i)
    {
        for (int j = 0; j < n2; ++j)
        {
            if(_idx1.at<int>(i) == _idx2.at<int>(j))
            {
                _inter1.push_back(i);
                _inter2.push_back(j);
                _inter_ori.push_back(_idx1.at<int>(i));
            }
        }
    }

    inter1.create(1, (int)_inter1.size(), CV_32S);
    inter2.create(1, (int)_inter2.size(), CV_32S);
    inter_ori.create(1, (int)_inter_ori.size(), CV_32S);

    for (int i = 0; i < (int)_inter1.size(); ++i)
    {
        inter1.getMat().at<int>(i) = _inter1[i];
    }
    for (int i = 0; i < (int)_inter2.size(); ++i)
    {
        inter2.getMat().at<int>(i) = _inter2[i];
    }
    for (int i = 0; i < (int)_inter_ori.size(); ++i)
    {
        inter_ori.getMat().at<int>(i) = _inter_ori[i];
    }
}