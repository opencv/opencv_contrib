#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>

struct PoseRow {
    double ts = 0.0;
    cv::Mat R; // 3x3
    cv::Mat t; // 3x1
};

static bool readTum(const std::string &path, std::vector<PoseRow> &out){
    std::ifstream ifs(path);
    if(!ifs.is_open()) return false;
    std::string line;
    while(std::getline(ifs, line)){
        if(line.empty() || line[0] == '#') continue;
        for(char &c : line) if(c == ',') c = ' '; // allow comma-separated CSV
        std::stringstream ss(line);
        double ts, tx, ty, tz, qx, qy, qz, qw;
        if(!(ss >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) continue;
        if(ts > 1e12) ts *= 1e-9; // convert nanoseconds to seconds when needed
        cv::Mat R = cv::Mat::eye(3,3,CV_64F);
        double xx = qx*qx, yy = qy*qy, zz = qz*qz;
        double xy = qx*qy, xz = qx*qz, yz = qy*qz;
        double wx = qw*qx, wy = qw*qy, wz = qw*qz;
        R.at<double>(0,0) = 1.0 - 2.0*(yy + zz);
        R.at<double>(0,1) = 2.0*(xy - wz);
        R.at<double>(0,2) = 2.0*(xz + wy);
        R.at<double>(1,0) = 2.0*(xy + wz);
        R.at<double>(1,1) = 1.0 - 2.0*(xx + zz);
        R.at<double>(1,2) = 2.0*(yz - wx);
        R.at<double>(2,0) = 2.0*(xz - wy);
        R.at<double>(2,1) = 2.0*(yz + wx);
        R.at<double>(2,2) = 1.0 - 2.0*(xx + yy);
        cv::Mat t = (cv::Mat_<double>(3,1) << tx, ty, tz);
        out.push_back({ts, R, t});
    }
    return true;
}

static void buildCorrespondence(const std::vector<PoseRow> &gt, const std::vector<PoseRow> &est,
                                std::vector<cv::Point3d> &p_gt, std::vector<cv::Point3d> &p_est){
    size_t i = 0, j = 0;
    while(i < gt.size() && j < est.size()){
        double ts_gt = gt[i].ts;
        double ts_est = est[j].ts;
            double diff = std::abs(ts_gt - ts_est);
            const double tol = 5e-2; // 50ms tolerance (EuRoC frames ~20Hz, ~0.05s step)
        if(diff < tol){
            p_gt.emplace_back(gt[i].t.at<double>(0), gt[i].t.at<double>(1), gt[i].t.at<double>(2));
            p_est.emplace_back(est[j].t.at<double>(0), est[j].t.at<double>(1), est[j].t.at<double>(2));
            i++; j++;
        } else if(ts_gt < ts_est){
            i++;
        } else {
            j++;
        }
    }
}

static void umeyamaSim3(const std::vector<cv::Point3d> &src, const std::vector<cv::Point3d> &dst,
                        double &scale, cv::Mat &R, cv::Mat &t){
    const int N = static_cast<int>(src.size());
    CV_Assert(N > 0 && dst.size() == src.size());
    cv::Mat src_mat(N, 3, CV_64F), dst_mat(N, 3, CV_64F);
    for(int i=0;i<N;++i){
        src_mat.at<double>(i,0) = src[i].x;
        src_mat.at<double>(i,1) = src[i].y;
        src_mat.at<double>(i,2) = src[i].z;
        dst_mat.at<double>(i,0) = dst[i].x;
        dst_mat.at<double>(i,1) = dst[i].y;
        dst_mat.at<double>(i,2) = dst[i].z;
    }
    cv::Mat mu_src, mu_dst;
    cv::reduce(src_mat, mu_src, 0, cv::REDUCE_AVG);
    cv::reduce(dst_mat, mu_dst, 0, cv::REDUCE_AVG);
    cv::Mat src_centered = src_mat - cv::repeat(mu_src, N, 1);
    cv::Mat dst_centered = dst_mat - cv::repeat(mu_dst, N, 1);
    cv::Mat Sigma = (dst_centered.t() * src_centered) / static_cast<double>(N);
    cv::SVD svd(Sigma);
    cv::Mat U = svd.u, Vt = svd.vt;
    cv::Mat S = cv::Mat::eye(3,3,CV_64F);
    if(cv::determinant(U * Vt) < 0) S.at<double>(2,2) = -1.0;
    R = U * S * Vt;
    double var_src = 0.0;
    for(int i=0;i<N;++i){
        cv::Vec3d v(src_centered.at<double>(i,0), src_centered.at<double>(i,1), src_centered.at<double>(i,2));
        var_src += v.dot(v);
    }
    var_src /= static_cast<double>(N);
    scale = cv::trace(R * Sigma)[0] / var_src;
    cv::Mat mu_src_t = mu_src.reshape(1,3);
    cv::Mat mu_dst_t = mu_dst.reshape(1,3);
    t = mu_dst_t - scale * R * mu_src_t;
}

static double computeATE(const std::vector<PoseRow> &gt, const std::vector<PoseRow> &est,
                         double &scale_out){
    std::vector<cv::Point3d> p_gt, p_est;
    buildCorrespondence(gt, est, p_gt, p_est);
    if(p_gt.size() < 2) return -1.0;
    cv::Mat R, t;
    double s = 1.0;
    umeyamaSim3(p_est, p_gt, s, R, t);
    scale_out = s;
    double se = 0.0;
    const size_t N = p_gt.size();
    for(size_t i=0;i<N;++i){
        cv::Mat p = (cv::Mat_<double>(3,1) << p_est[i].x, p_est[i].y, p_est[i].z);
        cv::Mat q = (cv::Mat_<double>(3,1) << p_gt[i].x, p_gt[i].y, p_gt[i].z);
        cv::Mat pe = s * R * p + t;
        cv::Mat d = pe - q;
        double dn = cv::norm(d);
        se += dn * dn;
    }
    return std::sqrt(se / static_cast<double>(N));
}

static double computeRPE(const std::vector<PoseRow> &gt, const std::vector<PoseRow> &est,
                         const cv::Mat &R_align, const cv::Mat &t_align, double scale, int delta){
    std::vector<cv::Point3d> p_gt, p_est;
    buildCorrespondence(gt, est, p_gt, p_est);
    if(p_gt.size() <= static_cast<size_t>(delta)) return -1.0;
    double se = 0.0;
    size_t count = 0;
    for(size_t i=0; i + delta < p_gt.size(); ++i){
        cv::Mat pe1 = (cv::Mat_<double>(3,1) << p_est[i].x, p_est[i].y, p_est[i].z);
        cv::Mat pe2 = (cv::Mat_<double>(3,1) << p_est[i+delta].x, p_est[i+delta].y, p_est[i+delta].z);
        cv::Mat pa1 = scale * R_align * pe1 + t_align;
        cv::Mat pa2 = scale * R_align * pe2 + t_align;
        cv::Point3d dg = p_gt[i+delta] - p_gt[i];
        cv::Point3d de(pa2.at<double>(0) - pa1.at<double>(0),
                       pa2.at<double>(1) - pa1.at<double>(1),
                       pa2.at<double>(2) - pa1.at<double>(2));
        double diff = cv::norm(dg - de);
        se += diff * diff;
        count++;
    }
    if(count == 0) return -1.0;
    return std::sqrt(se / static_cast<double>(count));
}

int main(int argc, char** argv){
    if(argc < 3){
        std::cout << "Usage: " << argv[0] << " [gt_tum.csv] [est_tum.csv]\n";
        return 1;
    }
    std::vector<PoseRow> gt, est;
    if(!readTum(argv[1], gt)){
        std::cerr << "Failed to read GT: " << argv[1] << std::endl;
        return 1;
    }
    if(!readTum(argv[2], est)){
        std::cerr << "Failed to read EST: " << argv[2] << std::endl;
        return 1;
    }
    double scale = 1.0;
    double ate = computeATE(gt, est, scale);
    cv::Mat R_align, t_align;
    // Recompute alignment to reuse for RPE
    {
        std::vector<cv::Point3d> p_gt, p_est;
        buildCorrespondence(gt, est, p_gt, p_est);
        if(!p_gt.empty()){
            umeyamaSim3(p_est, p_gt, scale, R_align, t_align);
        }
    }
    double rpe = computeRPE(gt, est, R_align, t_align, scale, 1);
    std::cout << "ATE_RMSE " << ate << "\n";
    std::cout << "RPE_RMSE(delta=1) " << rpe << "\n";
    std::cout << "Scale " << scale << "\n";
    return 0;
}
