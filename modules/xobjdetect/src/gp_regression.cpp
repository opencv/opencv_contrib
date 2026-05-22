// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/******************************************************************************
 *
 * This software is the core implementation of the fine-grained search (FGS) method
 * based on the Gaussian process regression with latent variables.
 *
 * Details of the method can be found in the follow paper
 *   Yuting Zhang, Kihyuk Sohn, Ruben Villegas, Gang Pan, Honglak Lee
 *   ``Improving Object Detection with Deep Convolutional Networks via Bayesian Optimization and Structured Prediction''
 *   IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Boston, MA, 2015.
 *
 * The following people contributed to the code:
 *   Yuting Zhang   <zyt.cse@gmail.com>
 *   Kihyuk Sohn    <kihyuks@umich.edu>
 *   Ruben Villegas <rubville@umich.edu>
 *   Xinchen Yan    <xcyan@umich.edu>
 *   Gang Pan       <gpan@zju.edu.cn>
 *   Honglak Lee    <honglak@eecs.umich.edu>
 *
*******************************************************************************/

#include <string>
#include <cmath>
#include "opencv2/xobjdetect/gp_regression.hpp"

typedef cv2::fgs_gp_box_reg BoxReg;
/** protected/private functions */
cv::Mat BoxReg::sgp_kNp1_forward(const cv::Mat & psiNp1, const cv::Mat & PsiN, const GPmodel_dfn & GPmodel) {
    int n = PsiN.cols;    // psiNp1: 4 x 1; PsiN: 4 x N
    double eta = GPmodel.normCov;
    cv::Mat D = PsiN - cv::repeat(psiNp1, 1, n);    // D: 4 x N

    cv::Mat Ds = D.mul(cv::repeat(GPmodel.diagSqrtLambda, 1, n));
    cv::Mat Ds2 = Ds.mul(Ds);    // sumDs2: 1 x N

    cv::Mat sumDs2 = cv::Mat::zeros(1, n, CV_64F);
    cv::reduce(Ds2, sumDs2, 0, cv::REDUCE_SUM);    // expsumDs2: 1 x N

    cv::Mat expsumDs2 = cv::Mat::zeros(1, n, CV_64F);
    cv::exp(-0.5*sumDs2, expsumDs2);

    cv::Mat kNp1 = eta * expsumDs2;     // kNp1: N x 1
    kNp1 = kNp1.t();
    return kNp1;
}

cv::Mat BoxReg::sgp_kNp1_backward(const cv::Mat & psiNp1, const cv::Mat & PsiN, const GPmodel_dfn & GPmodel) {
    int n = PsiN.cols;    // psiNp1: 4 x 1; PsiN: 4 x N
    double eta = GPmodel.normCov;
    cv::Mat D = PsiN - cv::repeat(psiNp1, 1, n);    // D: 4 x N

    cv::Mat Ds = D.mul(cv::repeat(GPmodel.diagSqrtLambda, 1, n));
    cv::Mat Ds2 = Ds.mul(Ds);    // sumDs2: 1 x N

    cv::Mat sumDs2 = cv::Mat::zeros(1, n, CV_64F);
    cv::reduce(Ds2, sumDs2, 0, cv::REDUCE_SUM);    // expsumDs2: 1 x N

    cv::Mat expsumDs2 = cv::Mat::zeros(1, n, CV_64F);
    cv::exp(-0.5*sumDs2, expsumDs2);

    cv::Mat kNp1 = eta * expsumDs2;     // kNp1: N x 1
    kNp1 = kNp1.t();

    // dkNp1_dpsiNp1 = bsxfun(@times, D.*repmat(kNp1.', size(D,1), 1), vec(GPmodel.diagSqrtLambda).^2);
    cv::Mat kNp1_t = kNp1.t();
    cv::Mat dkNp1_dpsiNp1 = D.mul(cv::repeat(kNp1_t, 4, 1)).mul(cv::repeat(GPmodel.diagSqrtLambda.mul(GPmodel.diagSqrtLambda), 1, n));
    // dkNp1_dpsiNp1: 4 x N
    return dkNp1_dpsiNp1;
}

double BoxReg::sgp_posterior_mu_forward(const cv::Mat & kNp1, const cv::Mat & KN, const cv::Mat & fN, const GPmodel_dfn & GPmodel) {
    // dmu_dkNp1: N x 1
    cv::Mat dmu_dkNp1 = KN.inv()*(fN-GPmodel.m0);
    double mu = GPmodel.m0 + kNp1.dot(dmu_dkNp1);
    return mu;
}
cv::Mat BoxReg::sgp_posterior_mu_backward(const cv::Mat &, const cv::Mat & KN, const cv::Mat & fN, const GPmodel_dfn & GPmodel) {
    cv::Mat dmu_dkNp1 = KN.inv()*(fN-GPmodel.m0);
    return dmu_dkNp1;
}

double BoxReg::sgp_posterior_s2_forward(const cv::Mat & kNp1, const cv::Mat & KN, const GPmodel_dfn & GPmodel) {
    double eta = GPmodel.normCov;
    // t: N x 1
    cv::Mat t = KN.inv() * kNp1;
    double s2 = eta + GPmodel.noiseSigma2 - kNp1.dot(t);
    return s2;
}
cv::Mat BoxReg::sgp_posterior_s2_backward(const cv::Mat & kNp1, const cv::Mat & KN, const GPmodel_dfn &) {
    cv::Mat t = KN.inv() * kNp1;
    cv::Mat ds2_dkNp1 = -2 * t;
    return ds2_dkNp1;
}

double BoxReg::normpdf(const double & x) {
    return 1/sqrt(2*M_PI)*exp(-x*x/2);
}

double BoxReg::normcdf(const double & x) {
    if (x<-5)
        return 0;
    else if (x>5)
        return 1;
    else {
        double cdf = 0;
        for (double t = -5; t < x; t = t + 0.01) {
            cdf = cdf + normpdf(t) * 0.01;
        }
        return cdf;
    }
}

double BoxReg::erf(const double & x) {
    if (x<-5)
        return -1;
    else if (x>5)
        return 1;
    else if (x>0) {
        double _erf = 0;
        for (double t = 0; t < x; t = t + 0.01)
            _erf = _erf + 0.01 * 2/sqrt(M_PI) * exp(-t*t);
        return _erf;
    } else {
        double _erf = 0;
        for (double t = x; t <= 0; t = t + 0.01)
            _erf = _erf - 0.01 * 2/sqrt(M_PI) * exp(-t*t);
        return _erf;
    }
}
double BoxReg::erfc(const double & x) {
    return 1 - erf(x);
}

double BoxReg::sgp_ei_forward(const double & mu, const double & s2, const double & fN_hat) {
    double s = sqrt(s2);
    double g = (mu - fN_hat)/s;
    double a = s * (g*normcdf(g) + normpdf(g));
    return a;
}

std::vector<double> BoxReg::sgp_ei_backward(const double & mu, const double & s2, const double & fN_hat) {
    std::vector<double> dadm_ds2;
    double s = sqrt(s2);

    double dadm = 0.5*erfc( (fN_hat - mu)/ sqrt(2) * s);
    double dads2 = 0.5 * normpdf((fN_hat-mu)/s);
    dadm_ds2.push_back(dadm);
    dadm_ds2.push_back(dads2);
    return dadm_ds2;
}

cv::Mat BoxReg::sgp_KN(const cv::Mat & PsiN, const GPmodel_dfn & GPmodel) {
    cv::Mat KN = sgp_cov_forward(GPmodel, 0, PsiN);
    return KN;
}


cv::Mat BoxReg::pdist(const cv::Mat & inputMat) {
    int n = inputMat.rows;
    cv::Mat outputMat = cv::Mat::zeros(n*(n-1)/2, 1, CV_64F);
    int k = 0;
    for (int i = 0; i < n; i++)
        for (int j = i+1; j < n; j++) {
            cv::Mat diff = inputMat.row(i) - inputMat.row(j);
            outputMat.at<double>(k) = sqrt(diff.dot(diff));
            k = k + 1;
        }
    return outputMat;
}

cv::Mat BoxReg::squareform(const cv::Mat & inputMat) {
    int n = (int)(floor(sqrt(inputMat.rows*2))+1);
    int k = 0;
    cv::Mat outputMat = cv::Mat::zeros(n, n, CV_64F);
    for (int i = 0; i < n; i++)
        for (int j = i+1; j < n; j++) {
            outputMat.at<double>(i,j) = inputMat.at<double>(k);
            outputMat.at<double>(j,i) = inputMat.at<double>(k);
            k = k + 1;
        }
    return outputMat;
}

cv::Mat BoxReg::sgp_cov_forward(const GPmodel_dfn & GPmodel, const double & z, const cv::Mat & Psi1) {
    int n = Psi1.cols; // Psi1: 4 x N
    cv::Mat expz = cv::Mat::zeros(4,1, CV_64F);
    cv::exp(-z*GPmodel.idxbScaleEnabled, expz);
    cv::Mat dsl = GPmodel.diagSqrtLambda.mul(expz); // 4 x 1
    // pairwise distance: D = pdist(bsxfun(@times, Psi1.', dsl));
    cv::Mat D = pdist(Psi1.mul(cv::repeat(dsl, 1, n)).t());
    // covariance matrix without noise
    // D: n(n-1)/2 x 1
    cv::Mat expD2 = cv::Mat::zeros(n*(n-1)/2, 1, CV_64F);
    cv::exp(-0.5*(D.mul(D)), expD2);
    cv::Mat Cv = GPmodel.normCov * expD2;
    cv::Mat C = squareform(Cv);
    C = C + GPmodel.normCov*cv::Mat::eye(n, n, CV_64F);
    // covariance matrix with noise
    cv::Mat Cnoisy = C + GPmodel.noiseSigma2 * cv::Mat::eye(n, n, CV_64F);
    return Cnoisy;
}
cv::Mat BoxReg::sgp_cov_backward(const GPmodel_dfn & GPmodel, const double & z, const cv::Mat & Psi1) {
    int n = Psi1.cols; // Psi1: 4 x N
    cv::Mat expz = cv::Mat::zeros(4,1, CV_64F);
    cv::exp(-z*GPmodel.idxbScaleEnabled, expz);
    cv::Mat dsl = GPmodel.diagSqrtLambda.mul(expz); // 4 x 1
    // pairwise distance: D = pdist(bsxfun(@times, Psi1.', dsl));
    cv::Mat D = pdist(Psi1.mul(cv::repeat(dsl, 1, n)).t());
    // covariance matrix without noise
    // D: n(n-1)/2 x 1
    cv::Mat expD2 = cv::Mat::zeros(n*(n-1)/2, 1, CV_64F);
    cv::exp(-0.5*(D.mul(D)), expD2);
    cv::Mat Cv = GPmodel.normCov * expD2;

    cv::Mat shl = (GPmodel.diagSqrtLambda).mul(GPmodel.idxbScaleEnabled);
    cv::Mat Psi1_z = Psi1.mul(GPmodel.idxbScaleEnabled);
    cv::Mat Dse = pdist(Psi1_z.mul(cv::repeat(shl, 1, n)).t()); // n(n-1)/2 x 1
    cv::Mat dC_dz_v = exp(-2*z) * Cv.mul(Dse.mul(Dse));
    cv::Mat dC_dz = squareform(dC_dz_v);
    return dC_dz_v;
}

double BoxReg::sgp_neg_acquisition_ei_forward(const GPmodel_dfn & GPmodel, const cv::Mat & psiNp1,
        const cv::Mat & PsiN, const cv::Mat & fN, const double & fN_hat, const cv::Mat & KN) {

    cv::Mat kNp1 = sgp_kNp1_forward(psiNp1, PsiN, GPmodel);
    double mu = sgp_posterior_mu_forward(kNp1, KN, fN, GPmodel);
    double s2 = sgp_posterior_s2_forward(kNp1, KN, GPmodel);
    double a = sgp_ei_forward(mu, s2, fN_hat);
    a = -a;
    return a;
}

cv::Mat BoxReg::sgp_neg_acquisition_ei_backward(const GPmodel_dfn & GPmodel, const cv::Mat & psiNp1,
        const cv::Mat & PsiN, const cv::Mat & fN, const double & fN_hat, const cv::Mat & KN) {

    cv::Mat kNp1 = sgp_kNp1_forward(psiNp1, PsiN, GPmodel);
    double mu = sgp_posterior_mu_forward(kNp1, KN, fN, GPmodel);
    double s2 = sgp_posterior_s2_forward(kNp1, KN, GPmodel);

    cv::Mat dkNp1_dpsiNp1 = sgp_kNp1_backward(psiNp1, PsiN, GPmodel);// 4 x N
    cv::Mat dmu_dNp1 = sgp_posterior_mu_backward(kNp1, KN, fN, GPmodel);// N x 1
    cv::Mat ds2_dNp1 = sgp_posterior_s2_backward(kNp1, KN, GPmodel);// N x 1
    std::vector<double> dadmu_ds2 = sgp_ei_backward(mu, s2, fN_hat);

    double da_dmu = dadmu_ds2[0];
    double da_ds2 = dadmu_ds2[1];

    cv::Mat da_psiNp1 = dkNp1_dpsiNp1 * (dmu_dNp1 * da_dmu + ds2_dNp1 * da_ds2);
    da_psiNp1 = -da_psiNp1;

    return da_psiNp1;
}

double BoxReg::sgp_negloglik_forward(const GPmodel_dfn & GPmodel, const double & z, const cv::Mat & Psi1, const cv::Mat & f) {
    cv::Mat KN = sgp_cov_forward(GPmodel, z, Psi1);
    double ll = sgp_negloglik_givenC_forward(GPmodel, KN, f);
    return ll;
}
double BoxReg::sgp_negloglik_backward(const GPmodel_dfn & GPmodel, const double & z, const cv::Mat & Psi1, const cv::Mat & f) {
    cv::Mat KN = sgp_cov_forward(GPmodel, z, Psi1);
    cv::Mat dKN_dz = sgp_cov_backward(GPmodel, z, Psi1);
    cv::Mat dll_dKN  = sgp_negloglik_givenC_backward(GPmodel, KN, f);

    cv::Mat dKN_dz_flatten = dKN_dz.reshape(0, dKN_dz.rows * dKN_dz.cols);
    cv::Mat dll_dKN_flatten = dll_dKN.reshape(0, dll_dKN.rows * dll_dKN.cols);
    double dll_dz = dKN_dz_flatten.dot(dll_dKN_flatten);
    return dll_dz;
}

double BoxReg::sgp_negloglik_givenC_forward(const GPmodel_dfn & GPmodel, const cv::Mat & C, const cv::Mat & f) {
    int n = f.rows;
    cv::Mat fn = f - GPmodel.m0;
    double fCf = (fn.t()*C.inv()).dot(fn.t());
    double ll = 0.5 * (log(pow((2*M_PI),n) * cv::determinant(C)) + fCf);
    return ll;
}

cv::Mat BoxReg::sgp_negloglik_givenC_backward(const GPmodel_dfn & GPmodel, const cv::Mat & C, const cv::Mat & f) {
    cv::Mat fn = f - GPmodel.m0;
    // dll_dCnoisy = -0.5 * ((inv(C)*fn)*(fn'*inv(C)) - inv(C)): n x n
    cv::Mat dll_dCnoisy =-0.5 * ((C.inv() * fn.t()) * (fn.t() * C.inv()) - C.inv());
    return dll_dCnoisy;
}

cv::Mat BoxReg::rect2cvmat(cv::Rect & rect) {
    cv::Mat cvmat = cv::Mat::zeros(1, 4, CV_64F);
    cvmat.at<double>(0) = rect.x + rect.width / 2;
    cvmat.at<double>(1) = rect.y + rect.height / 2;
    cvmat.at<double>(2) = log(rect.width);
    cvmat.at<double>(3) = log(rect.height);
    return cvmat;
}

cv::Rect BoxReg::cvmat2rect(cv::Mat & cvmat) {
    cv::Rect rect;
    rect.width = (int)exp(cvmat.at<double>(2));
    rect.height = (int)exp(cvmat.at<double>(3));
    rect.x = (int)(cvmat.at<double>(0) - rect.width / 2);
    rect.y = (int)(cvmat.at<double>(1) - rect.height / 2);
    return rect;
}

double BoxReg::findMinZ(const GPmodel_dfn & GPmodel, const double & z, const cv::Mat & PsiN1, const cv::Mat & fN) {
    double z_prev = z;
    double ll_prev = sgp_negloglik_forward(GPmodel, z_prev, PsiN1, fN);
    double dll_dz = sgp_negloglik_backward(GPmodel, z_prev, PsiN1, fN);
    double z_new = z_prev - dll_dz;
    double ll_new = sgp_negloglik_forward(GPmodel, z_new, PsiN1, fN);

    double invlr = 2;
    while (fabs(ll_new - ll_prev) > 1e-3) {
        ll_prev = ll_new;
        z_prev = z_new;
        dll_dz = sgp_negloglik_backward(GPmodel, z_prev, PsiN1, fN);
        z_new = z_prev - 1/invlr * dll_dz;
        ll_new = sgp_negloglik_forward(GPmodel, z_new, PsiN1, fN);
        invlr = invlr + 1;
    }

    return z_new;
}

cv::Mat BoxReg::findMinPsiNp1(const GPmodel_dfn & GPmodel, const cv::Mat & psiNp1,
        const cv::Mat & PsiN, const cv::Mat & fN, const double & fN_hat, const cv::Mat & KN) {
    cv::Mat psiNp1_prev = psiNp1;
    double aei_prev = sgp_neg_acquisition_ei_forward(GPmodel, psiNp1_prev, PsiN, fN, fN_hat, KN);
    cv::Mat da_psiNp1 = sgp_neg_acquisition_ei_backward(GPmodel, psiNp1_prev, PsiN, fN, fN_hat, KN);
    cv::Mat psiNp1_new = psiNp1_prev - da_psiNp1;
    double aei_new = sgp_neg_acquisition_ei_forward(GPmodel, psiNp1_new, PsiN, fN, fN_hat, KN);

    double invlr = 2;
    while (fabs(aei_new - aei_prev) > 1e-3) {
        aei_prev = aei_new;
        psiNp1_prev = psiNp1_new;
        da_psiNp1 = sgp_neg_acquisition_ei_backward(GPmodel, psiNp1_prev, PsiN, fN, fN_hat, KN);
        psiNp1_new = psiNp1_prev - 1/invlr * da_psiNp1;
        aei_new = sgp_neg_acquisition_ei_forward(GPmodel, psiNp1_new, PsiN, fN, fN_hat, KN);
        invlr = invlr + 1;
    }

    return psiNp1_new;
}

/** public functions */
void BoxReg::Load( const std::string & model_path ) {
    FILE * fin = fopen((const char *)model_path.c_str(), "r");
    double m0, nc, ns2;
    double dsl[4];
    int ise[4];

    if(!fscanf(fin, "%lf\n", &m0)) return;
    if(!fscanf(fin, "%lf %lf %lf %lf\n", &dsl[0], &dsl[1], &dsl[2], &dsl[3])) return;
    if(!fscanf(fin, "%lf\n", &nc)) return;
    if(!fscanf(fin, "%lf\n", &ns2)) return;
    if(!fscanf(fin, "%d %d %d %d\n", &ise[0], &ise[1], &ise[2], &ise[3])) return;

    fclose(fin);

    GPmodel_.m0 = m0;
    GPmodel_.normCov = nc;
    GPmodel_.noiseSigma2 = ns2;
    GPmodel_.diagSqrtLambda = cv::Mat::zeros(4, 1, CV_64F);
    GPmodel_.idxbScaleEnabled = cv::Mat::zeros(4, 1, CV_32S);
    for (int i = 0; i < 4; i++) {
        GPmodel_.diagSqrtLambda.at<double>(i) = dsl[i];
        GPmodel_.idxbScaleEnabled.at<int>(i) = ise[i];
    }
    return;
}

cv::Rect BoxReg::ProposeBox(std::vector< std::pair<cv::Rect, double> > known_boxes) {
    cv::Mat PsiN1; // 4 x N
    cv::Mat fN; // N x 1

    for (std::size_t i = 0; i < known_boxes.size(); i++) {
        cv::Rect rect = known_boxes[i].first;
        double score = known_boxes[i].second;
        cv::Mat cur_bbox = rect2cvmat(rect);
        PsiN1.push_back(cur_bbox);
        fN.push_back(score);
    }
    PsiN1 = PsiN1.t();

    //
    double fN_hat;
    cv::Point p_max;
    cv::minMaxLoc(fN, NULL, &fN_hat, NULL, &p_max);

    // TODO: check "z0 = anchorScales(j)"
    double z0 = 0;
    double z_hat = findMinZ(GPmodel_, z0, PsiN1, fN);

    double expnz = exp(-z_hat);

    cv::Mat PsiN = PsiN1; // 4 x N
    int n = PsiN.cols;
    PsiN = PsiN.mul(expnz*cv::repeat(GPmodel_.idxbScaleEnabled, 1, n));
    cv::Mat KN = sgp_cov_forward(GPmodel_, 0, PsiN);

    cv::Mat psiNp1_0 = PsiN.col(p_max.y);
    cv::Mat psiNp1_hat = findMinPsiNp1(GPmodel_, psiNp1_0, PsiN, fN, fN_hat, KN);
    cv::Mat psiNp1_hat_1 = psiNp1_hat;
    psiNp1_hat_1.mul(1/expnz*GPmodel_.idxbScaleEnabled);

    cv::Rect pbox = cvmat2rect(psiNp1_hat_1);
  return pbox;
}
