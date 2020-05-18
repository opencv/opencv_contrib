#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <opencv2/core/utility.hpp>
#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>

using namespace cv;
using namespace std;
#include<vector>
#include <string>

struct map_points
{
	vector<Point3d> position_3d;
	vector<vector<Vec3b> > colors;
};

class OrbSLAM
{
public:
	OrbSLAM(Mat calib_mat_, Mat dist_coeffs_);
	OrbSLAM();
	void alignImages(Mat& im1);
	vector<KeyPoint> gridFeatureDetect(Mat& img, Mat& descriptors, int grid);
	void getMatcheBF(Mat& descriptor1, Mat& descriptor2, vector<DMatch>& matches);
	void getMatcheKNN(Mat& descriptor1, Mat& descriptor2, vector<DMatch>& matches);
	double computeScore(Mat &M, vector<Point2f>& points1, vector<Point2f>& points2, const double T_M);
	bool reconstract(Mat& H, vector<Point2f>& points1,
		vector<Point2f>& points2);
	bool doMapInitialization(vector<Point2f>& points1, vector<Point2f>& points2);
	bool doTracking();
	void getImage(Mat &img)
	{
		frames.push_back(img);
		alignImages(frames.back());
	};
	void setCamera();
	void updateCovisibilityGraph();
protected:
	vector<Mat> gray_frames;
	vector<Mat> frames;
	vector<vector<KeyPoint> > keypoints;
	vector<Mat> descriptors;

	vector<vector<int> > covisibility_graph;
	vector<vector<Point3f> > cur_points3d;
	vector<vector<Point2f> > cur_points2d;
	vector<vector<Vec3b>>  colors;
	vector<Affine3d> cam_pose;
	Mat calib_mat;
	Mat dist_coeffs;
	Matx33d new_calib;
	int grid = 1;
};

OrbSLAM::OrbSLAM()
{
	covisibility_graph.resize(covisibility_graph.size() + 1);
	for (int i = 0; i < covisibility_graph.size(); i++)
	{
		covisibility_graph[i].resize(covisibility_graph.size());
	}
	covisibility_graph[covisibility_graph.size() - 1][covisibility_graph.size() - 1] = 0;
	setCamera();
};

OrbSLAM::OrbSLAM(Mat calib_mat_, Mat dist_coeffs_)
{
	/*
	covisibility_graph.resize(1);
	for(int i = 0; i < 1; i++)
		covisibility_graph[i].resize(1);
	covisibility_graph[0][0] = 0;
	*/
	calib_mat = calib_mat_;
	dist_coeffs = dist_coeffs_;
	new_calib = Matx33d(calib_mat);
	covisibility_graph.resize(covisibility_graph.size() + 1);
	for (int i = 0; i < covisibility_graph.size(); i++)
	{
		covisibility_graph[i].resize(covisibility_graph.size());
	}
	covisibility_graph[covisibility_graph.size() - 1][covisibility_graph.size() - 1] = 0;
};

//for logi 1080p HD C920
void OrbSLAM::setCamera()
{
	Mat calib_mat_(3, 3, CV_64FC1);
	Mat dist_coeffs_ = Mat::zeros(5, 1, CV_64FC1);
	/*
	calib_mat_.at<double>(0, 0) = 987;
	calib_mat_.at<double>(0, 1) = 0.0;
	calib_mat_.at<double>(0, 2) = 630;
	calib_mat_.at<double>(1, 0) = 0.0;
	calib_mat_.at<double>(1, 1) = 987;
	calib_mat_.at<double>(1, 2) = 357;
	calib_mat_.at<double>(2, 0) = 0.0;
	calib_mat_.at<double>(2, 1) = 0.0;
	calib_mat_.at<double>(2, 2) = 1.0;
	*/
	calib_mat_.at<double>(0, 0) = 612.03;
	calib_mat_.at<double>(0, 1) = 0.0;
	calib_mat_.at<double>(0, 2) = 320.15;
	calib_mat_.at<double>(1, 0) = 0.0;
	calib_mat_.at<double>(1, 1) = 661.6614;
	calib_mat_.at<double>(1, 2) = 117.5195;
	calib_mat_.at<double>(2, 0) = 0.0;
	calib_mat_.at<double>(2, 1) = 0.0;
	calib_mat_.at<double>(2, 2) = 1.0;
	calib_mat = calib_mat_;

	dist_coeffs_.at<double>(0, 0) = -0.128224;
	dist_coeffs_.at<double>(1, 0) = 0.023572;
	dist_coeffs_.at<double>(2, 0) = -0.0596;
	dist_coeffs_.at<double>(3, 0) = 0.040301;
	dist_coeffs_.at<double>(4, 0) = 0.0;
	dist_coeffs = dist_coeffs_;
	new_calib = Matx33d(612.03, 0.0, 320.15, 0.0, 661.6614, 117.5195, 0.0, 0.0, 1.0);
}

void OrbSLAM::updateCovisibilityGraph()
{
	covisibility_graph.resize(covisibility_graph.size() + 1);
	for (int i = 0; i < covisibility_graph.size(); i++)
	{
		covisibility_graph[i].resize(covisibility_graph.size());
	}
	covisibility_graph[covisibility_graph.size() - 1][covisibility_graph.size() - 1] = 0;
	colors.resize(colors.size() + 1);
	cur_points2d.resize(cur_points2d.size() + 1);
	vector<Point2f> points1, points2;
	for (size_t i = 0; i < covisibility_graph.size() - 1; i++)
	{
		int j = covisibility_graph.size() - 1;

		cout << "graph ver :" << i << " " << j << endl;

		vector<DMatch> matches;
		getMatcheBF(descriptors[i], descriptors[j], matches);
		covisibility_graph[i][j] = matches.size();
		//draw matches
		Mat imMatches;
		//drawMatches(frames[i], keypoints[i], frames[j], keypoints[j], matches, imMatches);
		//imshow("matches", imMatches);
		//waitKey();

		vector<KeyPoint> tmp_keypoints_i;
		vector<KeyPoint> tmp_keypoints_j;
		Mat tmp_descriptors_i(Size(0, 32), CV_8U);
		Mat tmp_descriptors_j(Size(0, 32), CV_8U);
		vector<size_t> ind_keypoints_i;
		vector<size_t> ind_keypoints_j;
		
		for (size_t n = 0; n < matches.size(); n++)
		{
			points1.push_back(keypoints[i][matches[n].queryIdx].pt);
			points2.push_back(keypoints[j][matches[n].trainIdx].pt);
			colors[colors.size() - 1].push_back(viz::Color(frames[j].at<Vec3b>(points2.back())));
			ind_keypoints_i.push_back(matches[n].queryIdx);
			ind_keypoints_j.push_back(matches[n].trainIdx);
		}
		//sort(ind_keypoints_i.begin(), ind_keypoints_i.end());
		//sort(ind_keypoints_j.begin(), ind_keypoints_j.end());
		for (size_t n = 0; n < keypoints[i].size(); n++)
		{
			for (size_t m = 0; m < ind_keypoints_i.size(); m++)
			{
				if (ind_keypoints_i[m] == n)
				{
					tmp_keypoints_i.push_back(keypoints[i][n]);
					tmp_descriptors_i.push_back(descriptors[i].row(n));
					break;
				}
			}
		}
		for (size_t n = 0; n < keypoints[j].size(); n++)
		{
			for (size_t m = 0; m < ind_keypoints_j.size(); m++)
			{
				if (ind_keypoints_j[m] == n)
				{
					tmp_keypoints_j.push_back(keypoints[j][n]);
					tmp_descriptors_j.push_back(descriptors[j].row(n));
					break;
				}
			}
		}
		descriptors[i].resize(tmp_descriptors_i.rows);
		descriptors[j].resize(tmp_descriptors_j.rows);
		descriptors[i] = tmp_descriptors_i;
		descriptors[j] = tmp_descriptors_j;
		keypoints[i] = tmp_keypoints_i;
		keypoints[j] = tmp_keypoints_j;
		


	}
	
	cur_points2d[cur_points2d.size() - 1].insert(cur_points2d[cur_points2d.size() - 1].end(),
		points2.cbegin(),
		points2.cend());
	if (points1.size() != 0)
		doMapInitialization(points1, points2);

}

bool OrbSLAM::doMapInitialization(vector<Point2f>& points1, vector<Point2f>& points2)
{
	//vector<Point2f> points1, points2;
	//alignImages(frames.back());
	cout << points1.size() << " " << points2.size() << endl;

	Mat h = findHomography(points1, points2, RANSAC, 5.99);
	Mat f(3, 3, CV_64FC1);
	f = findFundamentalMat(points1, points2, FM_RANSAC, 3.84);
	double score_f = computeScore(f, points1, points2, 3.84);
	double score_h = computeScore(h, points1, points2, 5.99);
	double R_h = score_h / (score_h + score_f);
	//vector<Point3d> points3d;
	if (R_h > 0.45)
	{
		return reconstract(h, points1, points2);
	}
	else
	{
		cout << "not implemented\n";
		// REWORK!!!!!!
		return reconstract(h, points1, points2);
	}
	return true;

}

void OrbSLAM::getMatcheKNN(Mat& descriptor1, Mat& descriptor2, vector<DMatch>& matches)
{
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
	std::vector< std::vector<DMatch> > knn_matches;
	matcher->knnMatch(descriptor1, descriptor2, knn_matches, 2);
	const float ratio_thresh = 0.6f;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i].size() >= 2)
		{
			if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
			{
				matches.push_back(knn_matches[i][0]);
			}
		}
	}
}

void OrbSLAM::getMatcheBF(Mat& descriptor1, Mat& descriptor2, vector<DMatch>& matches)
{
	float good_match_percent = 0.08f;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptor1, descriptor2, matches, Mat());
	std::sort(matches.begin(), matches.end(),
		[](DMatch a, DMatch b) { return a.distance > b.distance; });
	const int numGoodMatches = matches.size() * good_match_percent;
	matches.erase(matches.begin() + numGoodMatches, matches.end());

}

void OrbSLAM::alignImages(Mat& im1)
{
	Mat new_gray_frame;
	cvtColor(im1, new_gray_frame, cv::COLOR_BGR2GRAY);
	gray_frames.push_back(new_gray_frame);
	//keypoints.resize(keypoints.size() + 1);
	descriptors.resize(descriptors.size() + 1);
	//colors.resize(colors.size() + 1)
	cvtColor(im1, gray_frames.back(), cv::COLOR_BGR2GRAY);
	vector<KeyPoint> new_keypoints = gridFeatureDetect(gray_frames.back(), descriptors.back(), grid);

	keypoints.push_back(new_keypoints);
	cout << "keypoints size " << keypoints.size() << endl;
	//getMatcheBF(descriptors1, descriptors2, matches.back());

	//draw matches
	//Mat imMatches;
	//drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
	//imshow("matches", imMatches);
	//waitKey();
	//for( size_t i = 0; i < matches.back().size(); i++ )
	//{
	//    points1.push_back( keypoints1[ matches[i].queryIdx ].pt );
	//    points2.push_back( keypoints2[ matches[i].trainIdx ].pt );
	//    colors.push_back(viz::Color(im1.at<Vec3b>(points1[i])));
	//}

}
vector<KeyPoint> OrbSLAM::gridFeatureDetect(Mat& img, Mat& descriptor, int grid)
{
	vector<KeyPoint> keypoints_;
	int step_c = img.cols / grid;

	int step_r = img.rows / grid;

	int max_threshold = 500;
	int step_threshold = 8;
	int max_features = 4000;
	max_features = max_features / grid;
	vector<Mat> masks;
	for (int i = 0; i < img.cols; i += step_c)
	{
		for (int j = 0; j < img.rows; j += step_r)
		{
			Mat mask = Mat::zeros(img.size(), CV_8UC1);
			rectangle(mask, Point(i, j), Point(i + step_c, j + step_r), Scalar(255), -1);
			masks.push_back(mask);
		}
	}

	for (size_t i = 0; i < masks.size(); i++)
	{
		Ptr<ORB> orb = ORB::create(max_features);
		orb->setScoreType(ORB::FAST_SCORE);
		for (int threshold = 20; threshold < max_threshold; threshold += step_threshold)
		{
			vector<KeyPoint> keypoints_temp;
			Mat descriptors_temp;
			orb->setFastThreshold(threshold);
			orb->detectAndCompute(img, masks[i], keypoints_temp, descriptors_temp);
			if (keypoints_temp.size() >= 5)
			{
				std::copy(begin(keypoints_temp), end(keypoints_temp), std::back_inserter(keypoints_));
				descriptor.push_back(descriptors_temp);
				break;
			}
		}
	}
	return keypoints_;
}

//S = summ(p_M(d^2(m1, M^(-1) * m2) + p_M(d^2(m2, M * m1))))
//p_M(d^2) = 5.99 - d^2 if d^2 < 5.99
//else p_M(d^2) = 0
double OrbSLAM::computeScore(Mat &M, vector<Point2f>& points1, vector<Point2f>& points2, const double T_M)
{
	Mat M_inv = M.inv();
	Mat m2(3, points2.size(), CV_64FC1);
	for (int i = 0; i < points2.size(); i++)
	{
		m2.at<double>(0, i) = points2[i].x;
		m2.at<double>(1, i) = points2[i].y;
		m2.at<double>(2, i) = 1;
	}
	Mat M_inv_m2_mat = M_inv * m2;
	vector<Point2f> M_inv_m2;
	vector<double> dist1;
	for (int i = 0; i < points1.size(); i++)
	{
		M_inv_m2.push_back(Point2f(M_inv_m2_mat.at<double>(0, i) / M_inv_m2_mat.at<double>(2, i),
			M_inv_m2_mat.at<double>(1, i) / M_inv_m2_mat.at<double>(2, i)));
		dist1.push_back((M_inv_m2[i].x - points1[i].x) * (M_inv_m2[i].x - points1[i].x) +
			(M_inv_m2[i].y - points1[i].y) * (M_inv_m2[i].y - points1[i].y));
	}
	//TODO use convertPointsToHomogeneous() and convertPointsFromHomogeneous()
	Mat m1(3, points1.size(), CV_64FC1);
	for (int i = 0; i < points1.size(); i++)
	{
		m1.at<double>(0, i) = points1[i].x;
		m1.at<double>(1, i) = points1[i].y;
		m1.at<double>(2, i) = 0;
	}
	Mat M_m1_mat = M * m1;
	vector<Point2f> M_m1;
	vector<double> dist2;
	double S_M = 0;
	for (int i = 0; i < points2.size(); i++)
	{
		M_m1.push_back(Point2f(M_m1_mat.at<double>(0, i) / M_m1_mat.at<double>(2, i),
			M_m1_mat.at<double>(1, i) / M_m1_mat.at<double>(2, i)));
		dist2.push_back((M_m1[i].x - points2[i].x) * (M_m1[i].x - points2[i].x) +
			(M_m1[i].y - points2[i].y) * (M_m1[i].y - points2[i].y));
	}
	double T_H = 5.99;
	for (int i = 0; i < dist1.size(); i++)
	{
		if (dist1[i] < T_M)
			S_M += T_H - dist1[i];
		if (dist2[i] < T_M)
			S_M += T_H - dist2[i];
	}
	cout << "S_M = " << S_M << endl;
	return S_M;
}

bool OrbSLAM::reconstract(Mat& H, vector<Point2f>& points1, vector<Point2f>& points2)
{
	vector<Mat> rotations, translations, normals;
	int solutions = decomposeHomographyMat(H, calib_mat, rotations, translations, normals);
	cout << "solution " << solutions << endl;
	if (solutions == 0)
	{
		cur_points2d.resize(cur_points2d.size() - 1);
		colors.resize(colors.size() - 1);
		return false;
	}
	vector<int> possible_solution;
	filterHomographyDecompByVisibleRefpoints(rotations, normals, points1, points2, possible_solution);

	//cout << "possible solution " << possible_solution.size() << endl;
	if (possible_solution.size() == 0)
	{
		cur_points2d.resize(cur_points2d.size() - 1);
		return false;
	}
	vector<vector<Point3f> > possible_3d_points(possible_solution.size());
	vector<float> reprojection_error(possible_solution.size());
	for (int m = 0; m < possible_solution.size(); m++)
	{
		Mat projection1, projection2;
		Mat id_mat = Mat::eye(rotations[m].size(), CV_64FC1);
		Mat z = Mat::zeros(translations[m].size(), CV_64FC1);

		cv::sfm::projectionFromKRt(calib_mat, id_mat, z, projection1);
		cv::sfm::projectionFromKRt(calib_mat, rotations[possible_solution[m]], translations[possible_solution[m]], projection2);
		cv::Mat points1Mat(2, points1.size(), CV_64FC1);
		cv::Mat points2Mat(2, points1.size(), CV_64FC1);


		for (int i = 0; i < points1.size(); i++)
		{
			cv::Point2f myPoint1 = points1.at(i);
			cv::Point2f myPoint2 = points2.at(i);
			points1Mat.at<double>(0, i) = myPoint1.x;
			points1Mat.at<double>(1, i) = myPoint1.y;
			points2Mat.at<double>(0, i) = myPoint2.x;
			points2Mat.at<double>(1, i) = myPoint2.y;
		}

		vector<Mat> points2d;
		points2d.push_back(points1Mat);
		points2d.push_back(points2Mat);

		vector<Mat> projections;
		projections.push_back(projection1);
		projections.push_back(projection2);

		Mat points3d_mat(3, 1, CV_64FC1);
		Mat Rs;
		Mat Ts;
		cv::sfm::triangulatePoints(points2d, projections, points3d_mat);


		for (int i = 0; i < points3d_mat.cols; i++)
		{
			possible_3d_points[m].push_back(Point3f(points3d_mat.at<double>(0, i),
				points3d_mat.at<double>(1, i),
				points3d_mat.at<double>(2, i)));
		}

		vector<Point2f> new_points;
		projectPoints(possible_3d_points[m], rotations[possible_solution[m]], translations[possible_solution[m]], calib_mat, dist_coeffs, new_points);
		sort(new_points.begin(), new_points.end(),
			[](Point2f& a, Point2f& b) {return sqrt(a.x * a.x + a.y * a.y) <
			sqrt(b.x * b.x + b.y * b.y);    });
		sort(cur_points2d.back().begin(), cur_points2d.back().end(),
			[](Point2f& a, Point2f& b) {return sqrt(a.x * a.x + a.y * a.y) <
			sqrt(b.x * b.x + b.y * b.y);    });

		reprojection_error[m] = 0;
		for (int i = 0; i < new_points.size(); i++)
			reprojection_error[m] += sqrt(((new_points[i].x - cur_points2d.back()[i].x) * (new_points[i].x - cur_points2d.back()[i].x)) +
			((new_points[i].y - cur_points2d.back()[i].y) * (new_points[i].y - cur_points2d.back()[i].y)));
		cout << reprojection_error[m] << endl;
	}
	cout << "possible solution: " << reprojection_error.size() << endl;
	size_t ind_min_reprojection_err = 0;
	float min = numeric_limits<float>::max();
	for (size_t i = 0; i < reprojection_error.size(); i++)
	{
		if (reprojection_error[i] < min)
		{
			min = reprojection_error[i];
			ind_min_reprojection_err = i;
		}
	}
	cur_points3d.resize(cur_points3d.size() + 1);
	cur_points3d[cur_points3d.size() - 1] = possible_3d_points[ind_min_reprojection_err];
	doTracking();
	return true;
}

bool OrbSLAM::doTracking()
{
	cout << "in tracking";
	Mat rvec(Mat::zeros(3, 1, CV_64FC1));       // output rotation vector
	Mat tvec(Mat::zeros(3, 1, CV_64FC1));       // output translation vector
	Mat _R_matrix(Mat::zeros(3, 1, CV_64FC1)); // rotation matrix
	Mat _t_matrix(Mat::zeros(3, 1, CV_64FC1)); // translation matrix



	int iterationsCount = 500;        // number of Ransac iterations.
	float reprojectionError = 5.0;    // maximum allowed distance to consider it an inlier.
	float confidence = 0.95;          // RANSAC successful confidence.
	bool useExtrinsicGuess = false;

	int flags = 0;
	Mat inliers(Mat::zeros(1, 1, CV_64FC1));
	cout << "3d and 2d sizes:" << cur_points3d.back().size() << " " << cur_points2d.back().size() << endl;
	solvePnPRansac(cur_points3d.back(), cur_points2d.back(), calib_mat, dist_coeffs, rvec, tvec,
		useExtrinsicGuess, iterationsCount, reprojectionError);



	Rodrigues(rvec, _R_matrix);                   // converts Rotation Vector to Matrix
	_t_matrix = tvec;                            // set translation matrix

	cam_pose.push_back(Affine3d(_R_matrix, _t_matrix));
	

	vector<Point3d> all_points;
	vector<Vec3b> all_colors;

	for(int i = 0; i < cur_points3d.size(); i++)
		for(int j = 0; j < cur_points3d[i].size(); j++)
		{
			all_points.push_back(cur_points3d[i][j]);
			all_colors.push_back(colors[i][j]);
		}
	cout << "start visualization" << endl;
	viz::Viz3d window;
	//viz::WCameraPosition cpw(0.25); // Coordinate axes

	//viz::WCameraPosition cpw_frustum(new_calib, 0.3, viz::Color::yellow());
	//window.showWidget("coordinate", viz::WCoordinateSystem());
//    window.showWidget("CPW", cpw, cam_pose);
//used for visualization

	viz::WCloud cloud_wid(all_points, all_colors);
	cloud_wid.setRenderingProperty( cv::viz::POINT_SIZE, 5 );
	window.showWidget("cameras_frames_and_lines", viz::WTrajectory(cam_pose, viz::WTrajectory::BOTH, 0.1, viz::Color::green()));
	window.showWidget("cameras_frustums", viz::WTrajectoryFrustums(cam_pose, Matx33d(calib_mat), 0.1, viz::Color::yellow()));
	//window.showWidget("CPW_FRUSTUM", cpw_frustum, cam_pose);
	window.showWidget("points", cloud_wid);
	window.setViewerPose(cam_pose.back());
	window.spin();
	return true;
}