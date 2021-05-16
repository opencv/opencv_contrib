#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>

#include <iostream>
#include <vector>

#include "sba.h"

using namespace cv;

class Vslam
{
public:
	Vslam(Mat camera_matrix, Mat dist_coeffs);
	bool initMap(const Mat& img1, const Mat& img2);
	bool tracking(const Mat& img1, const Mat& img2);
	void localMapping();

protected:
	void detectFeatures(const Mat& img, std::vector<KeyPoint>& features, Mat& descriptors);
	bool matchFeatures(const Mat& descriptor1, const Mat& descriptor2,
					   std::vector<DMatch>& matches,
					   std::vector<int>& matched_ind1, std::vector<int>& matched_ind2,
					   std::vector<int>& unmatched_ind1, std::vector<int>& unmatched_ind2);
	double computeScore(const Mat &M,
						const std::vector<Point2f>& points1,
						const std::vector<Point2f>& points2,
						const double T_M, size_t num_iter);
	Mat chooseModel(const std::vector<Point2f>& points1, const std::vector<Point2f>& points2);
	bool RTFromHomography(const Mat& H,
						  const std::vector<Point2f>& points1, const std::vector<Point2f>& points2,
						  std::vector<Mat>& rotations, std::vector<Mat>& translations);
	bool RTFromFundamental(const Mat& F,
						   const std::vector<Point2f>& points1, const std::vector<Point2f>& points2,
						   std::vector<Mat>& rotations, std::vector<Mat>& translations);

	int motionChooseSolution(const std::vector<Point2f>& points1, const std::vector<Point2f>& points2,
							 const std::vector<Mat>& rotations, const std::vector<Mat>& translations,
							 std::vector<Point3f>& points_3f);
	void visualizeStructureAndMotion(const std::vector<Point3f>& points_3f,
									 const std::vector<Mat>& rotations,
									 const std::vector<Mat>& translations) const;
private:
	Mat camera_matrix, dist_coeffs;
	enum tForm { HOMOGRAPHY, FUNDAMENTAL } t_form;
	bool is_map_init = false;
	size_t frames_number = 0;

	std::vector<KeyPoint> prev_features, curr_features;

	std::vector<std::vector<KeyPoint> > unmatched_features;
	std::vector<Mat> unmatched_descriptors;

	Mat prev_descriptors, curr_descriptors;

	std::vector<Mat> R, t;
	std::vector<std::vector<Point3f> > point_3f_frame;

};

Vslam::Vslam(Mat camera_matrix_, Mat dist_coeffs_)
{
	camera_matrix = camera_matrix_.clone();
	dist_coeffs = dist_coeffs_.clone();
}

bool Vslam::initMap(const Mat& img1, const Mat& img2)
{
	if ((!is_map_init) && 
		(!img1.empty()) &&
		(!img2.empty()))
	{
		std::vector<KeyPoint> prev_features_tmp, curr_features_tmp;
		Mat prev_descriptors_tmp, curr_descriptors_tmp;

		detectFeatures(img1, prev_features_tmp, prev_descriptors_tmp);
		detectFeatures(img2, curr_features_tmp, curr_descriptors_tmp);

		std::vector<DMatch> matches;
		std::vector<int> matched_ind1, matched_ind2, unmatched_ind1, unmatched_ind2;
		
		if (!matchFeatures(prev_descriptors_tmp, curr_descriptors_tmp, matches,
						   matched_ind1, matched_ind2, 
						   unmatched_ind1, unmatched_ind2))
		{
			std::cout << "Not enough matches for initialization" << std::endl;
			return false;
		}

		prev_features.resize(matches.size());
		curr_features.resize(matches.size());

		std::vector<Point2f> points1(matches.size());
		std::vector<Point2f> points2(matches.size());

		Mat descriptors2(Size(32, matches.size()), CV_8U);
		for (size_t i = 0; i < matched_ind1.size(); i++)
		{
			int i1 = matched_ind1[i];
			int i2 = matched_ind2[i];

			prev_features[i] = prev_features_tmp[i1];
			curr_features[i] = curr_features_tmp[i2];

			curr_descriptors_tmp.row(i2).copyTo(descriptors2.row(i));

			points1[i] = prev_features[i].pt;
			points2[i] = curr_features[i].pt;
		}
		curr_descriptors = descriptors2.clone();
		/*
		unmatched_features.resize(2);
		unmatched_descriptors.resize(2);

		Mat descriptors1_tmp(Size(32, unmatched_ind1.size()), CV_8U);
		Mat descriptors2_tmp(Size(32, unmatched_ind2.size()), CV_8U);

		for (size_t i = 0; i < unmatched_ind1.size(); i++)
		{
			int i1 = unmatched_ind1[i];
			int i2 = unmatched_ind2[i];
			unmatched_features[0].push_back(prev_features_tmp[i1]);
			unmatched_features[1].push_back(prev_features_tmp[i2]);

			prev_descriptors_tmp.row(i1).copyTo(descriptors1_tmp.row(i));
			curr_descriptors_tmp.row(i2).copyTo(descriptors2_tmp.row(i));
		}
		unmatched_descriptors[0] = descriptors1_tmp.clone();
		unmatched_descriptors[1] = descriptors2_tmp.clone();
		*/
		Mat tform = chooseModel(points1, points2);

		std::cout << "t_form is " << t_form << std::endl;

		std::vector<Mat> rotations, translations;
		if (t_form == HOMOGRAPHY)
			RTFromHomography(tform, points1, points2, rotations, translations);
		else 
			RTFromFundamental(tform, points1, points2, rotations, translations);

		std::vector<Point3f> points_3f;
		int solution = motionChooseSolution(points1, points2, rotations, translations, points_3f);

		R.push_back(rotations[solution]);
		t.push_back(translations[solution]);
		visualizeStructureAndMotion(points_3f, R, t);
		point_3f_frame.push_back(points_3f);
		frames_number += 2;
		is_map_init = true;
		return true;
	}
	else if (is_map_init)
	{
		std::cout << "Map is already init" << std::endl;
		return false;
	}
	else
	{
		std::cout << "Need two images to start initialization" << std::endl;
		return false;
	}
}

bool Vslam::tracking(const Mat& img1, const Mat& img2)
{
	if ((is_map_init) && (!img2.empty()))
	{
		std::vector<KeyPoint> prev_features_tmp = curr_features;
		prev_descriptors = curr_descriptors.clone();

		std::vector<KeyPoint> curr_features_tmp;
		Mat curr_descriptors_tmp;

		detectFeatures(img2, curr_features_tmp, curr_descriptors_tmp);
	
		std::vector<DMatch> matches;
		std::vector<int> matched_ind1, matched_ind2, unmatched_ind1, unmatched_ind2;

		if (!matchFeatures(prev_descriptors, curr_descriptors_tmp, matches,
						   matched_ind1, matched_ind2, unmatched_ind1, unmatched_ind2))
		{
			return false;
		}
		//drawing
		Mat imMatches;
		drawMatches(img1, prev_features_tmp, img2, curr_features_tmp, matches, imMatches);
		imshow("matches2", imMatches);
		waitKey();

		prev_features.resize(matches.size());
		curr_features.resize(matches.size());

		std::vector<Point2f> points1(matches.size());
		std::vector<Point2f> points2(matches.size());

		Mat descriptors2(Size(32, matches.size()), CV_8U);

		for (size_t m = 0; m < matches.size(); m++)
		{
			int i1 = matches[m].queryIdx;
			int i2 = matches[m].trainIdx;

			prev_features[m] = prev_features_tmp[i1];
			curr_features[m] = curr_features_tmp[i1];

			curr_descriptors_tmp.row(i2).copyTo(descriptors2.row(m));

			points1[m] = prev_features[m].pt;
			points2[m] = curr_features[m].pt;
		}
		curr_descriptors = descriptors2.clone();

		Mat rvec(Mat::zeros(3, 1, CV_64F));   
		Mat tvec(Mat::zeros(3, 1, CV_64F));    

		solvePnPRansac(point_3f_frame[frames_number - 1], points2, camera_matrix, dist_coeffs, rvec, tvec);

		Mat rmatrix;
		Rodrigues(rvec, rmatrix);
		R.push_back(rmatrix);
		t.push_back(tvec);

		frames_number++;
		
		return true;
	}
	else if (!is_map_init)
	{
		std::cout << "Map is not init" << std::endl;
		return false;
	}
	else if (img2.empty())
	{
		std::cout << "Empty input image" << std::endl;
		return false;
	}
}

void Vslam::localMapping()
{

}

void Vslam::detectFeatures(const Mat& img, std::vector<KeyPoint>& features, Mat& descriptors)
{
	Mat img_gray;
	cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

	int max_features = 2000;
	Ptr<Feature2D> orb = ORB::create(max_features);
	orb->detectAndCompute(img_gray, Mat(), features, descriptors);

}

bool Vslam::matchFeatures(const Mat& descriptor1, const Mat& descriptor2, 
						  std::vector<DMatch>& matches,
						  std::vector<int>& matched_ind1, std::vector<int>& matched_ind2,
	                      std::vector<int>& unmatched_ind1, std::vector<int>& unmatched_ind2)
{
	float good_match_percent = 0.1;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptor1, descriptor2, matches, Mat());
	std::sort(matches.begin(), matches.end(),
			  [](DMatch a, DMatch b) { return a.distance > b.distance; });
	std::cout << matches.size() << std::endl;
	const int num_good_matches = matches.size() * good_match_percent;

	matched_ind1.resize(num_good_matches);
	matched_ind2.resize(num_good_matches);
	unmatched_ind1.resize(matches.size() - num_good_matches);
	unmatched_ind1.resize(matches.size() - num_good_matches);
	for (size_t m = 0; m < num_good_matches; m++)
	{
		matched_ind1[m] = matches[m].queryIdx;
		matched_ind2[m] = matches[m].trainIdx;
	}
	/*
	for (size_t m = num_good_matches; m < matches.size(); m++)
	{
		unmatched_ind1[m - num_good_matches] = matches[m].queryIdx;
		unmatched_ind2[m - num_good_matches] = matches[m].trainIdx;
	}
	*/
	/*
	const int min_matches_num = 100;
	if (num_good_matches < min_matches_num)
	{
		matches.clear();
		return false;
	}
	else
	{
		matches.erase(matches.begin() + num_good_matches, matches.end());
		return true;
	}
	*/
	matches.erase(matches.begin() + num_good_matches, matches.end());
	return true;
}


//S = summ(p_M(d^2(m1, M^(-1) * m2) + p_M(d^2(m2, M * m1))))
//p_M(d^2) = 5.99 - d^2 if d^2 < 5.99
//else p_M(d^2) = 0
double Vslam::computeScore(const Mat &M,
	const std::vector<Point2f>& points1,
	const std::vector<Point2f>& points2,
	const double T_M, size_t num_iter)
{
	Mat M_inv = M.inv();

	Mat m2(3, num_iter, CV_64F);
	if (points2.size() < num_iter)
		num_iter = points2.size();

	for (int i = 0; i < num_iter; i++)
	{
		m2.at<double>(0, i) = points2[i].x;
		m2.at<double>(1, i) = points2[i].y;
		m2.at<double>(2, i) = 1;
	}

	Mat M_inv_m2_mat = M_inv * m2;
	std::vector<Point2f> M_inv_m2;
	std::vector<double> dist1;
	for (int i = 0; i < num_iter; i++)
	{
		M_inv_m2.push_back(Point2f(M_inv_m2_mat.at<double>(0, i) / M_inv_m2_mat.at<double>(2, i),
			M_inv_m2_mat.at<double>(1, i) / M_inv_m2_mat.at<double>(2, i)));
		dist1.push_back((M_inv_m2[i].x - points1[i].x) * (M_inv_m2[i].x - points1[i].x) +
			(M_inv_m2[i].y - points1[i].y) * (M_inv_m2[i].y - points1[i].y));
	}
	//TODO: use convertPointsToHomogeneous() and convertPointsFromHomogeneous()
	Mat m1(3, num_iter, CV_64FC1);
	for (int i = 0; i < num_iter; i++)
	{
		m1.at<double>(0, i) = points1[i].x;
		m1.at<double>(1, i) = points1[i].y;
		m1.at<double>(2, i) = 0;
	}
	Mat M_m1_mat = M * m1;
	std::vector<Point2f> M_m1;
	std::vector<double> dist2;
	double S_M = 0;
	for (int i = 0; i < num_iter; i++)
	{
		M_m1.push_back(Point2f(M_m1_mat.at<double>(0, i) / M_m1_mat.at<double>(2, i),
			M_m1_mat.at<double>(1, i) / M_m1_mat.at<double>(2, i)));
		dist2.push_back((M_m1[i].x - points2[i].x) * (M_m1[i].x - points2[i].x) +
			(M_m1[i].y - points2[i].y) * (M_m1[i].y - points2[i].y));
	}
	double T_H = 5.99;
	for (int i = 0; i < num_iter; i++)
	{
		if (dist1[i] < T_M)
			S_M += T_H - dist1[i];
		if (dist2[i] < T_M)
			S_M += T_H - dist2[i];
	}
	std::cout << "S_M = " << S_M << std::endl;
	return S_M;
}

Mat Vslam::chooseModel(const std::vector<Point2f>& points1, const std::vector<Point2f>& points2)
{
	std::vector<uchar> mask;
	Mat h = findHomography(points1, points2, mask, RANSAC, 5.99);

	Mat f(3, 3, CV_64FC1);
	f = findFundamentalMat(points1, points2, FM_RANSAC, 3.84);

	double score_f = 0.0, score_h = 0.0, R_h;
	if(!f.empty())
		score_f = computeScore(f, points1, points2, 3.84, 8);
	if(!h.empty())
		score_h = computeScore(h, points1, points2, 5.99, 4);

	std::cout << "score_f is " << score_f << std::endl;
	std::cout << "score_h is " << score_h << std::endl;

	R_h = score_h / (score_h + score_f);
	if (R_h > 0.45)
	{
		t_form = HOMOGRAPHY;
		return h;
	}
	else
	{
		t_form = FUNDAMENTAL;
		return f;
	}
	
}

bool Vslam::RTFromHomography(const Mat& H, 
							 const std::vector<Point2f>& points1, const std::vector<Point2f>& points2, 
							 std::vector<Mat>& rotations, std::vector<Mat>& translations)
{
	std::vector<Mat> normals;
	int solutions = decomposeHomographyMat(H, camera_matrix, rotations, translations, normals);

	if (solutions == 0)
		return false;

	return true;
}

bool Vslam::RTFromFundamental(const Mat& F, 
							  const std::vector<Point2f>& points1, const std::vector<Point2f>& points2,
							  std::vector<Mat>& rotations, std::vector<Mat>& translations)
{
	Mat E = (F.t().mul(camera_matrix)).mul(F);
	Mat rotation1, rotation2;
	Mat mat_translations;
	decomposeEssentialMat(E, rotation1, rotation2, mat_translations);

	rotations.resize(4);
	rotations[0] = rotation1;
	rotations[1] = rotation1;
	rotations[2] = rotation2;
	rotations[3] = rotation2;

	Mat neg_mat_translation(3, 1, CV_64F);
	for(int i = 0; i < 3; i++)
		neg_mat_translation.at<double>(i, 0) = -mat_translations.at<double>(i, 0);

	translations.resize(4);
	translations[0] = mat_translations;
	translations[1] = neg_mat_translation;
	translations[2] = mat_translations;
	translations[3] = neg_mat_translation;

	return true;
}

int Vslam::motionChooseSolution(const std::vector<Point2f>& points1, const std::vector<Point2f>& points2,
								const std::vector<Mat>& rotations, const std::vector<Mat>& translations,
								std::vector<Point3f>& points_3f)
{
	std::vector<std::vector<Point3f> > points_3f_tmp(rotations.size());
	std::vector<int> count_visible_points(rotations.size());

	for (size_t i = 0; i < rotations.size(); i++)
	{
		Mat rotation_translation1, rotation_translation2;

		Mat identity_mat = Mat::eye(rotations[i].size(), CV_64F);
		Mat z = Mat::zeros(translations[i].size(), CV_64F);
		hconcat(identity_mat, z, rotation_translation1);
		Mat projection1 = camera_matrix * rotation_translation1;

		hconcat(rotations[i], translations[i], rotation_translation2);
		Mat projection2 = camera_matrix * rotation_translation2;

		Mat points_4f;
		triangulatePoints(projection1, projection2, points1, points2, points_4f);

		for (size_t k = 0; k < points_4f.cols; k++)
		{
			Mat X(3, 1, CV_32F);
			X.at<float>(0, 0) = points_4f.at<float>(0, k) / points_4f.at<float>(3, k);
			X.at<float>(1, 0) = points_4f.at<float>(1, k) / points_4f.at<float>(3, k);
			X.at<float>(2, 0) = points_4f.at<float>(2, k) / points_4f.at<float>(3, k);
			
			points_3f_tmp[i].push_back(Point3f(X.at<float>(0, 0),
											   X.at<float>(1, 0),
										       X.at<float>(2, 0)));

			double d1 = sfm::depth(identity_mat, z, X);
			double d2 = sfm::depth(rotations[i], translations[i], X);

			// Test if point is front to the two cameras.
			if (d1 > 0 && d2 > 0)
			{
				count_visible_points[i]++;
			}
		}
	}

	int solution_ind  = std::distance(count_visible_points.begin(), 
									  std::max_element(count_visible_points.begin(), count_visible_points.end()));
	points_3f = points_3f_tmp[solution_ind];

	return solution_ind;
}

void Vslam::visualizeStructureAndMotion(const std::vector<Point3f>& points_3f,
										const std::vector<Mat>& rotations, 
										const std::vector<Mat>& translations) const
{
	std::vector<Affine3f> cam_poses(rotations.size());
	for(size_t i = 0; i < rotations.size(); i++)
		cam_poses[i] = Affine3d(rotations[i], translations[i]);

	viz::Viz3d window;
	viz::WCloud cloud_wid(points_3f, viz::Color::green());
	cloud_wid.setRenderingProperty(cv::viz::POINT_SIZE, 2);
	window.showWidget("cameras_frames_and_lines", viz::WTrajectory(cam_poses, viz::WTrajectory::BOTH, 0.1, viz::Color::green()));
	window.showWidget("points", cloud_wid);
	window.setViewerPose(cam_poses.back());
	window.spin();
}