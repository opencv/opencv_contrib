#ifndef OPENCV_SAC_IMPL_HPP
#define OPENCV_SAC_IMPL_HPP

#include "opencv2/calib3d.hpp"
#include "opencv2/ptcloud/sac_segmentation.hpp"

namespace cv {
namespace ptcloud {

// testing & samples
CV_EXPORTS void generatePlane(Mat &cloud, const std::vector<double> &coeffs, int N=256);
CV_EXPORTS void generateSphere(Mat &cloud, const std::vector<double> &coeffs, int N=256);
CV_EXPORTS void generateCylinder(Mat &cloud, const std::vector<double> &coeffs, int N=256);
CV_EXPORTS void generateRandom(Mat &cloud, const std::vector<double> &coeffs, int N=256);


// for testing
struct CV_EXPORTS SPRT {
	virtual ~SPRT() {}
    virtual bool addDataPoint(int tested_point, double error) = 0;
    virtual bool isModelGood(int tested_point) = 0;
    virtual int update (int inlier_size) = 0;
    virtual SACScore getScore() = 0;

    static Ptr<SPRT> create(int state, int points_size_,
          double inlier_threshold_, double prob_pt_of_good_model, double prob_pt_of_bad_model,
          double time_sample, double avg_num_models, int sample_size_, int max_iterations_, ScoreMethod score_type_);
};

CV_EXPORTS Mat generateNormals(const Mat &cloud);
CV_EXPORTS std::vector<double> optimizeModel(int model_type, const Mat &cloud, const std::vector<int> &inliers_indices, const std::vector<double> &old_coeff);
CV_EXPORTS SACScore getInliers(int model_type, const std::vector<double> &coeffs, const Mat &cloud, const Mat &normals, const std::vector<int> &indices, const double threshold, const double best_inliers, std::vector<int>& inliers, double normal_distance_weight_, Ptr<SPRT> sprt=Ptr<SPRT>());
CV_EXPORTS bool generateHypothesis(int model_type, const Mat &cloud, const std::vector<int> &current_model_inliers, const Mat &normals, double max_radius, double normal_distance_weight_, std::vector<double> &coefficients);

} // ptcloud
} // cv

#endif // OPENCV_SAC_IMPL_HPP
