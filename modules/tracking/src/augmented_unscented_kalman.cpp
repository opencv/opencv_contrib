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
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
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

#include "precomp.hpp"
#include "opencv2/tracking/kalman_filters.hpp"

namespace cv {
namespace detail {
inline namespace tracking {
inline namespace kalman_filters {

void AugmentedUnscentedKalmanFilterParams::
    init( int dp, int mp, int cp, double processNoiseCovDiag, double measurementNoiseCovDiag,
                                Ptr<UkfSystemModel> dynamicalSystem, int type )
{
    CV_Assert( dp > 0 && mp > 0 );
    DP = dp;
    MP = mp;
    CP = std::max( cp, 0 );
    CV_Assert( type == CV_32F || type == CV_64F );
    dataType = type;

    this->model = dynamicalSystem;

    stateInit = Mat::zeros(DP, 1, type);
    errorCovInit = Mat::eye(DP, DP, type);

    processNoiseCov = processNoiseCovDiag*Mat::eye(DP, DP, type);
    measurementNoiseCov = measurementNoiseCovDiag*Mat::eye(MP, MP, type);

    alpha = 1e-3;
    k = 0.0;
    beta = 2.0;
}

AugmentedUnscentedKalmanFilterParams::
    AugmentedUnscentedKalmanFilterParams( int dp, int mp, int cp, double processNoiseCovDiag, double measurementNoiseCovDiag,
                                          Ptr<UkfSystemModel> dynamicalSystem, int type )
{
    init( dp, mp, cp, processNoiseCovDiag, measurementNoiseCovDiag, dynamicalSystem, type );
}


class AugmentedUnscentedKalmanFilterImpl: public UnscentedKalmanFilter
{

    int DP;                                     // dimensionality of the state vector
    int MP;                                     // dimensionality of the measurement vector
    int CP;                                     // dimensionality of the control vector
    int DAug;                                   // dimensionality of the augmented vector, DAug = 2*DP + MP
    int dataType;                               // type of elements of vectors and matrices

    Mat state;                                  // estimate of the system state (x*), DP x 1
    Mat errorCov;                               // estimate of the state cross-covariance matrix (P), DP x DP

    Mat stateAug;                               // augmented state vector (xa*), DAug x 1,
                                                // xa* = ( x*
                                                //         0
                                                //        ...
                                                //         0 )
    Mat errorCovAug;                            // estimate of the state cross-covariance matrix (Pa), DAug x DAug
                                                // Pa = (  P, 0, 0
                                                //         0, Q, 0
                                                //         0, 0, R  )

    Mat processNoiseCov;                        // process noise cross-covariance matrix (Q), DP x DP
    Mat measurementNoiseCov;                    // measurement noise cross-covariance matrix (R), MP x MP

    Ptr<UkfSystemModel> model;                  // object of the class containing functions for computing the next state and the measurement.

// Parameters of algorithm
    double alpha;                               // parameter, default is 1e-3
    double k;                                   // parameter, default is 0
    double beta;                                // parameter, default is 2.0

    double lambda;                              // internal parameter, lambda = alpha*alpha*( DP + k ) - DP;
    double tmpLambda;                           // internal parameter, tmpLambda = alpha*alpha*( DP + k );

// Auxillary members
    Mat measurementEstimate;                    // estimate of current measurement (y*), MP x 1

    Mat sigmaPoints;                            // set of sigma points ( x_i, i = 1..2*DP+1 ), DP x 2*DP+1

    Mat transitionSPFuncVals;                   // set of state function values at sigma points ( f_i, i = 1..2*DP+1 ), DP x 2*DP+1
    Mat measurementSPFuncVals;                  // set of measurement function values at sigma points ( h_i, i = 1..2*DP+1 ), MP x 2*DP+1

    Mat transitionSPFuncValsCenter;             // set of state function values at sigma points minus estimate of state ( fc_i, i = 1..2*DP+1 ), DP x 2*DP+1
    Mat measurementSPFuncValsCenter;            // set of measurement function values at sigma points minus estimate of measurement ( hc_i, i = 1..2*DP+1 ), MP x 2*DP+1

    Mat Wm;                                     // vector of weights for estimate mean, 2*DP+1 x 1
    Mat Wc;                                     // matrix of weights for estimate covariance, 2*DP+1 x 2*DP+1

    Mat gain;                                   // Kalman gain matrix (K), DP x MP
    Mat xyCov;                                  // estimate of the covariance between x* and y* (Sxy), DP x MP
    Mat yyCov;                                  // estimate of the y* cross-covariance matrix (Syy), MP x MP

    Mat r;                                      // zero vector of process noise for getting transitionSPFuncVals,
    Mat q;                                      // zero vector of measurement noise for getting measurementSPFuncVals

    Mat getSigmaPoints(const Mat& mean, const Mat& covMatrix, double coef);

public:

    AugmentedUnscentedKalmanFilterImpl(const AugmentedUnscentedKalmanFilterParams& params);
    ~AugmentedUnscentedKalmanFilterImpl() CV_OVERRIDE;

    Mat predict(InputArray control) CV_OVERRIDE;
    Mat correct(InputArray measurement) CV_OVERRIDE;

    Mat getProcessNoiseCov() const CV_OVERRIDE;
    Mat getMeasurementNoiseCov() const CV_OVERRIDE;
    Mat getErrorCov() const CV_OVERRIDE;

    Mat getState() const CV_OVERRIDE;

};

AugmentedUnscentedKalmanFilterImpl::AugmentedUnscentedKalmanFilterImpl(const AugmentedUnscentedKalmanFilterParams& params)
{
    alpha = params.alpha;
    beta = params.beta;
    k = params.k;

    CV_Assert( params.DP > 0 && params.MP > 0 );
    CV_Assert( params.dataType == CV_32F || params.dataType == CV_64F );
    DP = params.DP;
    MP = params.MP;
    CP = std::max( params.CP, 0 );
    dataType = params.dataType;

    DAug = DP + DP + MP;

    model = params.model;

    stateAug = Mat::zeros( DAug, 1, dataType );
    state = stateAug( Rect( 0, 0, 1, DP ));

    CV_Assert( params.stateInit.cols == 1 && params.stateInit.rows == DP );
    params.stateInit.copyTo(state);

    CV_Assert( params.processNoiseCov.cols == DP && params.processNoiseCov.rows == DP );
    CV_Assert( params.measurementNoiseCov.cols == MP && params.measurementNoiseCov.rows == MP );
    processNoiseCov = params.processNoiseCov.clone();
    measurementNoiseCov = params.measurementNoiseCov.clone();

    errorCovAug = Mat::zeros( DAug, DAug, dataType );
    errorCov = errorCovAug( Rect( 0, 0, DP, DP ) );
    Mat Q = errorCovAug( Rect( DP, DP, DP, DP ) );
    Mat R = errorCovAug( Rect( 2*DP, 2*DP, MP, MP ) );
    processNoiseCov.copyTo( Q );
    measurementNoiseCov.copyTo( R );

    CV_Assert( params.errorCovInit.cols == DP && params.errorCovInit.rows == DP );
    params.errorCovInit.copyTo( errorCov );

    measurementEstimate = Mat::zeros( MP, 1, dataType);

    gain = Mat::zeros( DAug, DAug, dataType );

    transitionSPFuncVals = Mat::zeros( DP, 2*DAug+1, dataType );
    measurementSPFuncVals = Mat::zeros( MP, 2*DAug+1, dataType );

    transitionSPFuncValsCenter = Mat::zeros( DP, 2*DAug+1, dataType );
    measurementSPFuncValsCenter = Mat::zeros( MP, 2*DAug+1, dataType );

    lambda = alpha*alpha*( DAug + k ) - DAug;
    tmpLambda = lambda + DAug;

    double tmp2Lambda = 0.5/tmpLambda;

    Wm = tmp2Lambda * Mat::ones( 2*DAug+1, 1, dataType );
    Wc = tmp2Lambda * Mat::eye( 2*DAug+1, 2*DAug+1, dataType );

    if ( dataType == CV_64F )
    {
        Wm.at<double>(0,0) = lambda/tmpLambda;
        Wc.at<double>(0,0) = lambda/tmpLambda + 1.0 - alpha*alpha + beta;
    }
    else
    {
        Wm.at<float>(0,0) = (float)(lambda/tmpLambda);
        Wc.at<float>(0,0) = (float)(lambda/tmpLambda + 1.0 - alpha*alpha + beta);
    }

}

AugmentedUnscentedKalmanFilterImpl::~AugmentedUnscentedKalmanFilterImpl()
{
    stateAug.release();
    errorCovAug.release();

    state.release();
    errorCov.release();

    processNoiseCov.release();
    measurementNoiseCov.release();

    measurementEstimate.release();

    sigmaPoints.release();

    transitionSPFuncVals.release();
    measurementSPFuncVals.release();

    transitionSPFuncValsCenter.release();
    measurementSPFuncValsCenter.release();

    Wm.release();
    Wc.release();

    gain.release();
    xyCov.release();
    yyCov.release();

    r.release();
    q.release();

}

Mat AugmentedUnscentedKalmanFilterImpl::getSigmaPoints(const Mat &mean, const Mat &covMatrix, double coef)
{
// x_0 = mean
// x_i = mean + coef * cholesky( covMatrix ), i = 1..n
// x_(i+n) = mean - coef * cholesky( covMatrix ), i = 1..n

    int n = mean.rows;
    Mat points = repeat(mean, 1, 2*n+1);

    Mat covMatrixL = covMatrix.clone();

// covMatrixL = cholesky( covMatrix )
    if ( dataType == CV_64F )
        choleskyDecomposition<double>(
                    covMatrix.ptr<double>(), covMatrix.step, covMatrix.rows,
                    covMatrixL.ptr<double>(), covMatrixL.step );
    else if ( dataType == CV_32F )
        choleskyDecomposition<float>(
                    covMatrix.ptr<float>(), covMatrix.step, covMatrix.rows,
                    covMatrixL.ptr<float>(), covMatrixL.step );

    covMatrixL = coef * covMatrixL;

    Mat p_plus = points( Rect( 1, 0, n, n ) );
    Mat p_minus = points( Rect( n+1, 0, n, n ) );

    add(p_plus, covMatrixL, p_plus);
    subtract(p_minus, covMatrixL, p_minus);

    return points;
}

Mat AugmentedUnscentedKalmanFilterImpl::predict(InputArray _control)
{
    Mat control = _control.getMat();
// get sigma points from xa* and Pa
    sigmaPoints = getSigmaPoints( stateAug, errorCovAug, sqrt( tmpLambda ) );

// compute f-function values at sigma points
// f_i = f(x_i[0:DP-1], control, x_i[DP:2*DP-1]), i = 0..2*DAug
    Mat x, fx;
    for ( int i = 0; i<2*DAug+1; i++)
    {
        x = sigmaPoints( Rect( i, 0, 1, DP) );
        q = sigmaPoints( Rect( i, DP, 1, DP) );
        fx = transitionSPFuncVals( Rect( i, 0, 1, DP) );
        model->stateConversionFunction( x, control, q, fx );
    }

// compute the estimate of state as mean f-function value at sigma point
// x* = SUM_{i=0}^{2*DAug}( Wm[i]*f_i )
    state = transitionSPFuncVals * Wm;

// compute f-function values at sigma points minus estimate of state
// fc_i = f_i - x*, i = 0..2*DAug
    subtract(transitionSPFuncVals, repeat( state, 1, 2*DAug+1 ), transitionSPFuncValsCenter);

// compute the estimate of the state cross-covariance matrix
// P = SUM_{i=0}^{2*DAug}( Wc[i]*fc_i*fc_i.t )
    errorCov = transitionSPFuncValsCenter * Wc * transitionSPFuncValsCenter.t();

    return state.clone();
}

Mat AugmentedUnscentedKalmanFilterImpl::correct(InputArray _measurement)
{
    Mat measurement = _measurement.getMat();
// get sigma points from xa* and Pa
    sigmaPoints = getSigmaPoints( stateAug, errorCovAug, sqrt( tmpLambda ) );

// compute h-function values at sigma points
// h_i = h(x_i[0:DP-1], x_i[2*DP:DAug-1]), i = 0..2*DAug
    Mat x, hx;
    measurementEstimate.setTo(0);
    for ( int i = 0; i<2*DAug+1; i++)
    {
        x = transitionSPFuncVals( Rect( i, 0, 1, DP) );
        r = sigmaPoints( Rect( i, 2*DP, 1, MP) );
        hx = measurementSPFuncVals( Rect( i, 0, 1, MP) );
        model->measurementFunction( x, r, hx );
    }

// compute the estimate of measurement as mean h-function value at sigma point
// y* = SUM_{i=0}^{2*DAug}( Wm[i]*h_i )
    measurementEstimate = measurementSPFuncVals * Wm;

// compute h-function values at sigma points minus estimate of state
// hc_i = h_i - y*, i = 0..2*DAug
    subtract(measurementSPFuncVals, repeat( measurementEstimate, 1, 2*DAug+1 ), measurementSPFuncValsCenter);

// compute the estimate of the y* cross-covariance matrix
// Syy = SUM_{i=0}^{2*DAug}( Wc[i]*hc_i*hc_i.t )
    yyCov = measurementSPFuncValsCenter * Wc * measurementSPFuncValsCenter.t();

// compute the estimate of the covariance between x* and y*
// Sxy = SUM_{i=0}^{2*DAug}( Wc[i]*fc_i*hc_i.t )
    xyCov = transitionSPFuncValsCenter * Wc * measurementSPFuncValsCenter.t();

// compute the Kalman gain matrix
// K = Sxy * Syy^(-1)
    gain = xyCov * yyCov.inv(DECOMP_SVD);

// compute the corrected estimate of state
// x* = x* + K*(y - y*), y - current measurement
    state = state + gain * ( measurement - measurementEstimate );

// compute the corrected estimate of the state cross-covariance matrix
// P = P - K*Sxy.t
    errorCov = errorCov - gain * xyCov.t();

    return state.clone();
}

Mat AugmentedUnscentedKalmanFilterImpl::getProcessNoiseCov() const
{
    return processNoiseCov.clone();
}

Mat AugmentedUnscentedKalmanFilterImpl::getMeasurementNoiseCov() const
{
    return measurementNoiseCov.clone();
}

Mat AugmentedUnscentedKalmanFilterImpl::getErrorCov() const
{
    return errorCov.clone();
}

Mat AugmentedUnscentedKalmanFilterImpl::getState() const
{
    return state.clone();
}

Ptr<UnscentedKalmanFilter> createAugmentedUnscentedKalmanFilter(const AugmentedUnscentedKalmanFilterParams &params)
{
    Ptr<UnscentedKalmanFilter> kfu( new AugmentedUnscentedKalmanFilterImpl(params) );
    return kfu;
}

}}}}  // namespace
