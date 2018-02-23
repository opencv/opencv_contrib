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

#include "test_precomp.hpp"
#include "opencv2/tracking/kalman_filters.hpp"

namespace opencv_test { namespace {
using namespace cv::tracking;

// In this two tests Augmented Unscented Kalman Filter are applied to the dynamic system from example "The reentry problem" from
// "A New Extension of the Kalman Filter to Nonlinear Systems" by Simon J. Julier and Jeffrey K. Uhlmann.
class BallisticModel: public UkfSystemModel
{
    static const double step_h;

    Mat diff_eq(const Mat& x)
    {
        double x1 = x.at<double>(0, 0);
        double x2 = x.at<double>(1, 0);
        double x3 = x.at<double>(2, 0);
        double x4 = x.at<double>(3, 0);
        double x5 = x.at<double>(4, 0);

        const double h0 = 9.3;
        const double beta0 = 0.59783;
        const double Gm = 3.9860044 * 1e5;
        const double r_e = 6374;

        const double r = sqrt( x1*x1 + x2*x2 );
        const double v = sqrt( x3*x3 + x4*x4 );
        const double d = - beta0 * exp( ( r_e - r )/h0 ) * exp( x5 ) * v;
        const double g = - Gm / (r*r*r);

        Mat fx = x.clone();

        fx.at<double>(0, 0) = x3;
        fx.at<double>(1, 0) = x4;
        fx.at<double>(2, 0) = d * x3 + g * x1;
        fx.at<double>(3, 0) = d * x4 + g * x2;
        fx.at<double>(4, 0) = 0.0;

        return fx;
    }
public:
    void stateConversionFunction(const Mat& x_k, const Mat& u_k, const Mat& v_k, Mat& x_kplus1)
    {
        Mat v = sqrt(step_h) * v_k.clone();
        v.at<double>(0, 0) = 0.0;
        v.at<double>(1, 0) = 0.0;

        Mat k1 = diff_eq( x_k ) + v;
        Mat tmp = x_k + step_h*0.5*k1;
        Mat k2 = diff_eq( tmp ) + v;
        tmp = x_k + step_h*0.5*k2;
        Mat k3 = diff_eq( tmp ) + v;
        tmp = x_k + step_h*k3;
        Mat k4 = diff_eq( tmp ) + v;

        x_kplus1 = x_k + (1.0/6.0)*step_h*( k1 + 2.0*k2 + 2.0*k3 + k4 ) + u_k;
    }

    void measurementFunction(const Mat& x_k, const Mat& n_k, Mat& z_k)
    {
        double x1 = x_k.at<double>(0, 0);
        double x2 = x_k.at<double>(1, 0);
        double x1_r = 6374.0;
        double x2_r = 0.0;

        double R = sqrt( pow( x1 - x1_r, 2 ) + pow( x2 - x2_r, 2 ) );
        double Phi = atan( (x2 - x2_r)/(x1 - x1_r) );

        R += n_k.at<double>(0, 0);
        Phi += n_k.at<double>(1, 0);

        z_k.at<double>(0, 0) = R;
        z_k.at<double>(1, 0) = Phi;
    }
};

const double BallisticModel::step_h = 0.05;

TEST(AUKF, br_landing_point)
{
    const double abs_error = 0.1;

    const int nIterations = 4000; // number of iterations before landing
    const double landing_coordinate = 2.5; // the expected landing coordinate

    const double alpha = 1;
    const double beta = 2.0;
    const double kappa = -2.0;

    int MP = 2;
    int DP = 5;
    int CP = 0;
    int type = CV_64F;

    Mat processNoiseCov = Mat::zeros( DP, DP, type );
    processNoiseCov.at<double>(0, 0) = 1e-14;
    processNoiseCov.at<double>(1, 1) = 1e-14;
    processNoiseCov.at<double>(2, 2) = 2.4065 * 1e-5;
    processNoiseCov.at<double>(3, 3) = 2.4065 * 1e-5;
    processNoiseCov.at<double>(4, 4) = 1e-6;
    Mat processNoiseCovSqrt = Mat::zeros( DP, DP, type );
    sqrt( processNoiseCov, processNoiseCovSqrt );

    Mat measurementNoiseCov = Mat::zeros( MP, MP, type );
    measurementNoiseCov.at<double>(0, 0) = 1e-3*1e-3;
    measurementNoiseCov.at<double>(1, 1) = 0.13*0.13;
    Mat measurementNoiseCovSqrt = Mat::zeros( MP, MP, type );
    sqrt( measurementNoiseCov, measurementNoiseCovSqrt );

    RNG rng( 117 );

    Mat state( DP, 1, type );
    state.at<double>(0, 0) = 6500.4;
    state.at<double>(1, 0) = 349.14;
    state.at<double>(2, 0) = -1.8093;
    state.at<double>(3, 0) = -6.7967;
    state.at<double>(4, 0) = 0.6932;

    Mat initState = state.clone();
    initState.at<double>(4, 0) = 0.0;

    Mat P = 1e-6 * Mat::eye( DP, DP, type );
    P.at<double>(4, 4) = 1.0;

    Mat measurement( MP, 1, type );

    Mat q( DP, 1, type );
    Mat r( MP, 1, type );

    Ptr<BallisticModel> model( new BallisticModel() );
    AugmentedUnscentedKalmanFilterParams params( DP, MP, CP, 0, 0, model );

    params.stateInit = initState.clone();
    params.errorCovInit = P.clone();
    params.measurementNoiseCov = measurementNoiseCov.clone();
    params.processNoiseCov = processNoiseCov.clone();

    params.alpha = alpha;
    params.beta = beta;
    params.k = kappa;

    Ptr<UnscentedKalmanFilter> augmentedUncsentedKalmanFilter = createAugmentedUnscentedKalmanFilter(params);

    Mat correctStateUKF( DP, 1, type );
    Mat u = Mat::zeros( DP, 1, type );

    for (int i = 0; i<nIterations; i++)
    {
        rng.fill( q, RNG::NORMAL, Scalar::all(0),  Scalar::all(1) );
        q = processNoiseCovSqrt*q;

        rng.fill( r, RNG::NORMAL, Scalar::all(0), Scalar::all(1) );
        r = measurementNoiseCovSqrt*r;

        model->stateConversionFunction(state, u, q, state);
        model->measurementFunction(state, r, measurement);

        augmentedUncsentedKalmanFilter->predict();
        correctStateUKF = augmentedUncsentedKalmanFilter->correct( measurement );
    }

    double landing_y = correctStateUKF.at<double>(1, 0);
    ASSERT_NEAR(landing_coordinate, landing_y, abs_error);
}

TEST(DISABLED_AUKF, DISABLED_br_mean_squared_error)
{
    const double velocity_treshold = 0.004;
    const double state_treshold = 0.04;

    const int nIterations = 4000; // number of iterations before landing

    const double alpha = 1;
    const double beta = 2.0;
    const double kappa = -2.0;

    int MP = 2;
    int DP = 5;
    int CP = 0;
    int type = CV_64F;

    Mat processNoiseCov = Mat::zeros( DP, DP, type );
    processNoiseCov.at<double>(0, 0) = 1e-14;
    processNoiseCov.at<double>(1, 1) = 1e-14;
    processNoiseCov.at<double>(2, 2) = 2.4065 * 1e-5;
    processNoiseCov.at<double>(3, 3) = 2.4065 * 1e-5;
    processNoiseCov.at<double>(4, 4) = 1e-6;
    Mat processNoiseCovSqrt = Mat::zeros( DP, DP, type );
    sqrt( processNoiseCov, processNoiseCovSqrt );

    Mat measurementNoiseCov = Mat::zeros( MP, MP, type );
    measurementNoiseCov.at<double>(0, 0) = 1e-3*1e-3;
    measurementNoiseCov.at<double>(1, 1) = 0.13*0.13;
    Mat measurementNoiseCovSqrt = Mat::zeros( MP, MP, type );
    sqrt( measurementNoiseCov, measurementNoiseCovSqrt );

    RNG rng( 464 );

    Mat state( DP, 1, type );
    state.at<double>(0, 0) = 6500.4;
    state.at<double>(1, 0) = 349.14;
    state.at<double>(2, 0) = -1.8093;
    state.at<double>(3, 0) = -6.7967;
    state.at<double>(4, 0) = 0.6932;

    Mat initState = state.clone();
    Mat initStateKF = state.clone();
    initStateKF.at<double>(4, 0) = 0.0;

    Mat P = 1e-6 * Mat::eye( DP, DP, type );
    P.at<double>(4, 4) = 1.0;

    Mat measurement( MP, 1, type );

    Mat q( DP, 1, type);
    Mat r( MP, 1, type);

    Ptr<BallisticModel> model( new BallisticModel() );
    AugmentedUnscentedKalmanFilterParams params( DP, MP, CP, 0, 0, model );

    params.stateInit = initStateKF.clone();
    params.errorCovInit = P.clone();
    params.measurementNoiseCov = measurementNoiseCov.clone();
    params.processNoiseCov = processNoiseCov.clone();

    params.alpha = alpha;
    params.beta = beta;
    params.k = kappa;

    Mat predictStateUKF( DP, 1, type );
    Mat correctStateUKF( DP, 1, type );

    Mat errors = Mat::zeros( nIterations, 4, type );
    Mat u = Mat::zeros( DP, 1, type );

    for (int j = 0; j<100; j++)
    {
        cv::Ptr<UnscentedKalmanFilter> augmentedUncsentedKalmanFilter = createAugmentedUnscentedKalmanFilter(params);
        state = initState.clone();

        for (int i = 0; i<nIterations; i++)
        {
            rng.fill( q, RNG::NORMAL, Scalar::all(0),  Scalar::all(1) );
            q = processNoiseCovSqrt*q;

            rng.fill( r, RNG::NORMAL, Scalar::all(0), Scalar::all(1) );
            r = measurementNoiseCovSqrt*r;

            model->stateConversionFunction(state, u, q, state);
            model->measurementFunction(state, r, measurement);

            predictStateUKF = augmentedUncsentedKalmanFilter->predict();
            correctStateUKF = augmentedUncsentedKalmanFilter->correct( measurement );

            Mat errorUKF = state - correctStateUKF;

            for (int l = 0; l<4; l++)
                errors.at<double>(i, l) += pow( errorUKF.at<double>(l, 0), 2.0 );

        }
    }

    errors = errors/100.0;
    sqrt( errors, errors );

    double max_x1 = cvtest::norm(errors.col(0), NORM_INF);
    double max_x2 = cvtest::norm(errors.col(1), NORM_INF);
    double max_x3 = cvtest::norm(errors.col(2), NORM_INF);
    double max_x4 = cvtest::norm(errors.col(3), NORM_INF);

    ASSERT_GE( state_treshold, max_x1 );
    ASSERT_GE( state_treshold, max_x2 );
    ASSERT_GE( velocity_treshold, max_x3 );
    ASSERT_GE( velocity_treshold, max_x4 );

}


// In this test Augmented Unscented Kalman Filter are applied to the univariate nonstationary growth model (UNGM).
// This model was used in example from "Unscented Kalman filtering for additive noise case: Augmented vs. non-augmented"
// by Yuanxin Wu and Dewen Hu.
class UnivariateNonstationaryGrowthModel: public UkfSystemModel
{

public:
    void stateConversionFunction(const Mat& x_k, const Mat& u_k, const Mat& v_k, Mat& x_kplus1)
    {
        double x = x_k.at<double>(0, 0);
        double n = u_k.at<double>(0, 0);
        double q = v_k.at<double>(0, 0);
        double u = u_k.at<double>(0, 0);

        double x1 = 0.5*x + 25*( x/(x*x + 1) ) + 8*cos( 1.2*(n-1) ) + q + u;
        x_kplus1.at<double>(0, 0) = x1;
    }
    void measurementFunction(const Mat& x_k, const Mat& n_k, Mat& z_k)
    {
        double x = x_k.at<double>(0, 0);
        double r = n_k.at<double>(0, 0);

        double y = x*x/20.0 + r;
        z_k.at<double>(0, 0) = y;
    }
};

TEST(AUKF, DISABLED_ungm_mean_squared_error)
{

    const double alpha = 1.5;
    const double beta = 2.0;
    const double kappa = 0.0;

    const double mse_treshold = 0.05;
    const int nIterations = 500; // number of observed iterations

    int MP = 1;
    int DP = 1;
    int CP = 0;
    int type = CV_64F;

    Ptr<UnivariateNonstationaryGrowthModel> model( new UnivariateNonstationaryGrowthModel() );
    AugmentedUnscentedKalmanFilterParams params( DP, MP, CP, 0, 0, model );

    Mat processNoiseCov = Mat::zeros( DP, DP, type );
    processNoiseCov.at<double>(0, 0) = 1.0;
    Mat processNoiseCovSqrt = Mat::zeros( DP, DP, type );
    sqrt( processNoiseCov, processNoiseCovSqrt );

    Mat measurementNoiseCov = Mat::zeros( MP, MP, type );
    measurementNoiseCov.at<double>(0, 0) = 1.0;
    Mat measurementNoiseCovSqrt = Mat::zeros( MP, MP, type );
    sqrt( measurementNoiseCov, measurementNoiseCovSqrt );

    Mat P = Mat::eye( DP, DP, type );

    Mat state( DP, 1, type );
    state.at<double>(0, 0) = 0.1;

    Mat initState = state.clone();
    initState.at<double>(0, 0) = 0.0;

    params.errorCovInit = P;
    params.measurementNoiseCov = measurementNoiseCov;
    params.processNoiseCov = processNoiseCov;
    params.stateInit = initState.clone();

    params.alpha = alpha;
    params.beta = beta;
    params.k = kappa;

    Mat correctStateAUKF( DP, 1, type );

    Mat measurement( MP, 1, type );
    Mat exactMeasurement( MP, 1, type );

    Mat q( DP, 1, type );
    Mat r( MP, 1, type );

    Mat u( DP, 1, type );
    Mat zero = Mat::zeros( MP, 1, type );

    RNG rng( 216 );

    double average_error = 0.0;
    for (int j = 0; j<1000; j++)
    {
        cv::Ptr<UnscentedKalmanFilter> augmentedUncsentedKalmanFilter = createAugmentedUnscentedKalmanFilter( params );
        state = params.stateInit.clone();

        double mse = 0.0;
        for (int i = 0; i<nIterations; i++)
        {
            rng.fill( q, RNG::NORMAL, Scalar::all(0), Scalar::all(1) );
            rng.fill( r, RNG::NORMAL, Scalar::all(0), Scalar::all(1) );
            q = processNoiseCovSqrt*q;
            r = measurementNoiseCovSqrt*r;

            u.at<double>(0, 0) = (double)i;
            model->stateConversionFunction(state, u, q, state);

            model->measurementFunction(state, zero, exactMeasurement);
            model->measurementFunction(state, r, measurement);

            augmentedUncsentedKalmanFilter->predict( u );
            correctStateAUKF = augmentedUncsentedKalmanFilter->correct( measurement );

            mse +=  pow( state.at<double>(0, 0) - correctStateAUKF.at<double>(0, 0), 2.0 );
        }
        mse /= nIterations;
        average_error += mse;
    }
    average_error /= 1000.0;

    ASSERT_GE( mse_treshold, average_error );
}

}} // namespace
