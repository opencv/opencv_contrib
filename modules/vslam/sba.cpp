#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <stdio.h>

using namespace cv;
#include <vector>

struct CameraParams
{
	double ppx = 0.0;
	double ppy = 0.0;
	double focal = 0.0;
	double aspect = 0.0;
	Mat camera_mat;

	CameraParams(const Mat& camera)
	{
		ppx = camera.at<double>(0, 2);
		ppy = camera.at<double>(1, 2);
		focal = camera.at<double>(0, 0);
		aspect = camera.at<double>(1, 1) / camera.at<double>(0, 0);
		camera_mat = camera.clone();
	}

	CameraParams() {};

	void setCameraParams(const Mat& camera)
	{
		ppx = camera.at<double>(0, 2);
		ppy = camera.at<double>(1, 2);
		focal = camera.at<double>(0, 0);
		aspect = camera.at<double>(1, 1) / camera.at<double>(0, 0);
		camera_mat = camera.clone();
	}

	Mat getCameraParamsMat() const
	{
		return camera_mat;
	}
};

struct Quaternion
{
	double w = 0.0;
	double x = 0.0;
	double y = 0.0;
	double z = 0.0;
	Quaternion(const std::vector<double>& q_vec)
	{
		x = q_vec[0];
		y = q_vec[1];
		z = q_vec[2];
		w = q_vec[3];
	}
};

struct Mat_xy
{
	Mat x;
	Mat y;
};

class LevMarqBA
{
public:
	LevMarqBA(std::vector<Point3f> points3d,
		std::vector<std::vector<Point2f> > points2d, //i=views
		std::vector<Mat> rotations_,
		std::vector<Mat> translations_,
		Mat camera_params_);
		//Mat dist_coeffs,
		//std::vector<std::vector<int> > visibility_indices_

	//protected:
	Point3f computeProjection(const Mat& camera_matrix,
		const Mat& rotation,
		const Mat& translation,
		const Point3f X);

	Mat computeErrors(const std::vector<std::vector<Point2f> >& X,
		const std::vector<std::vector<Point2f> >& X_hat);

	Mat computedfdR(const Point3f xyz_point,
		const Point3f uvw_point,
		const Point3f C,
		const Mat& R,
		const CameraParams& camera);

	Mat computedfdX(const Point3f xyz_point,
		const Point3f uvw_point,
		const Mat& R,
		const CameraParams& camera);

	Mat computedRdq(const Quaternion q);

	Mat computedfdq(const Point3f xyz_point,
		const Point3f uvw_point,
		const Point3f C,
		const Mat& R,
		const CameraParams& camera);

	Mat computedfdC(const Point3f xyz_point,
		const Point3f uvw_point,
		const Mat& R,
		const CameraParams& camera);

	Mat computeDiagCovarMatrix(const std::vector<std::vector<Point2f> >& X);

	Mat augmentateNonZeroElements(const Mat& A, const double mu);

	Mat computeJacobian(const std::vector<Point3f>& X_3d,
		const std::vector<std::vector<int> >& visibility_indices,
		const std::vector<Mat>& R,
		const std::vector<Mat>& t,
		const CameraParams& camera);

	void findDelta(const double mu);
	Quaternion RMatToQuaternion(Mat R);

private:
	CameraParams camera_params;
	std::vector<std::vector<Point2f> > X, X_hat;
	std::vector<Point3f> X_3d;
	std::vector<std::vector<int> > visibility_indices;
	std::vector<Mat> rotations, translations;

	size_t n_views, n_points;
	Mat A, B, J;
	//Mat diag_cov, diag_cov_inv;
	Mat U, V, W;
	Mat U_augm, V_augm;
	Mat Y;

	Mat errors;
	Mat errors_a, errors_b;

	/*
	Mat A_x, A_y, B_x, B_y, J_x, J_y;
	Mat U_x, U_y, V_x, V_y, W_x, W_y;
	Mat U_x_augm, U_y_augm, V_x_augm, V_y_augm;
	Mat Y_x, Y_y;
	Mat diag_cov_x, diag_cov_inv_x;
	Mat diag_cov_y, diag_cov_inv_y;

	Mat errors_x, errors_y;
	Mat errors_a_x, errors_a_y;
	Mat errors_b_x, errors_b_y;
	*/
};

LevMarqBA::LevMarqBA(std::vector<Point3f> points3d,
					std::vector<std::vector<Point2f> > points2d, 
					std::vector<Mat> rotations_,
					std::vector<Mat> translations_,
					Mat camera_params_)
{
	X_3d = points3d;
	X = points2d;

	rotations = rotations_;
	translations = translations_;

	camera_params.setCameraParams(camera_params_);

	n_views = rotations.size();
	n_points = X_3d.size();
}

Point3f LevMarqBA::computeProjection(const Mat& camera_matrix,
	const Mat& rotation,
	const Mat& translation,
	const Point3f X)
{
	Mat rot_transl;
	hconcat(rotation, translation, rot_transl);

	Mat X_mat(4, 1, CV_64F);
	X_mat.at<double>(0, 0) = X.x;
	X_mat.at<double>(1, 0) = X.y;
	X_mat.at<double>(2, 0) = X.z;
	X_mat.at<double>(3, 0) = 1.0;

	Mat uvw_mat = camera_matrix * rot_transl * X_mat;

	Point3f uvw_point(uvw_mat);
	return uvw_point;
}

Mat LevMarqBA::computeErrors(const std::vector<std::vector<Point2f> >& X,
	const std::vector<std::vector<Point2f> >& X_hat)
{
	Mat errs(n_views, n_points, CV_64F);

	for (size_t i = 0; i < errs.rows; i++)
	{
		for (size_t j = 0; j < errs.cols; j++)
		{
			if (X[i][j] != Point2f(-1.0, -1.0))
				errs.at<double>(j, i) = norm(X[i][j] - X_hat[i][j]);
			else
				errs.at<double>(j, i) = 0.0;
		}
	}
	return errs;
}

//TODO: input mat of 2f points
Mat LevMarqBA::computeDiagCovarMatrix(const std::vector<std::vector<Point2f> >& X)
{
	Mat cov, mean;

	calcCovarMatrix(X, cov, mean, COVAR_NORMAL | COVAR_COLS);

	int nsamples = n_views * n_points;
	cov = cov / (nsamples - 1);

	Mat diag_cov = Mat::zeros(2 * nsamples, 2 * nsamples, CV_64F);
	Mat identity_mat = Mat::ones(Size(2, 2), CV_64F);
	for (int i = 0; i < nsamples; i++)
	{
		Mat cov_block = cov.at<double>(i, i) * identity_mat;
		cov_block.copyTo(diag_cov(Rect(2 * i, 2 * i, 2, 2)));
	}
	return diag_cov;
}

/*
void LevMarqBA::computeDiagCovarMatrix(const std::vector<std::vector<Point2d> >& X)
{
	Mat X_mat;

	Mat(X).reshape(X.size()).convertTo(X_mat, CV_64F);
	std::vector<Mat> channels(2);
	split(X_mat, channels);

	//std::cout << "X mat = " << X_mat << std::endl;
	Mat cov_x, mean_x;
	Mat cov_y, mean_y;

	calcCovarMatrix(channels[0], cov_x, mean_x, COVAR_NORMAL | COVAR_COLS);
	calcCovarMatrix(channels[1], cov_y, mean_y, COVAR_NORMAL | COVAR_COLS);

	int nsamples = X_mat.cols;
	cov_x = cov_x / (nsamples - 1);
	cov_y = cov_y / (nsamples - 1);

	diag_cov_x = Mat::zeros(nsamples, nsamples, CV_64F);
	diag_cov_inv_x = diag_cov_x.clone();

	diag_cov_y = Mat::zeros(nsamples, nsamples, CV_64F);
	diag_cov_inv_y = diag_cov_y.clone();

	for (int i = 0; i < nsamples; i++)
	{
		diag_cov_x.at<double>(i, i) = cov_x.at<double>(i, i);
		diag_cov_inv_x.at<double>(i, i) = 1 / diag_cov_x.at<double>(i, i);

		diag_cov_y.at<double>(i, i) = cov_y.at<double>(i, i);
		diag_cov_inv_y.at<double>(i, i) = 1 / diag_cov_y.at<double>(i, i);
	}
	*/
	/*
	int nsamples = X.size();

	Mat X_mat(X);
	std::vector<Mat> channels(2);
	split(X_mat, channels);

	diag_cov_x = Mat::zeros(nsamples, nsamples, CV_64F);
	diag_cov_inv_x = diag_cov_x.clone();

	diag_cov_y = Mat::zeros(nsamples, nsamples, CV_64F);
	diag_cov_inv_y = diag_cov_y.clone();

	double mean_x = sum(mean(channels[0]))[0];
	double mean_y = sum(mean(channels[1]))[0];

	for (int i = 0; i < nsamples; i++)
	{
		double t_x = channels[0].at<double>(i, 0) - mean_x;
		diag_cov_x.at<double>(i, i) = t_x * t_x / (nsamples - 1);
		diag_cov_inv_x.at<double>(i, i) = 1 / diag_cov_x.at<double>(i, i);

		double t_y = channels[1].at<double>(i, 0) - mean_y;
		diag_cov_y.at<double>(i, i) = t_y * t_y / (nsamples - 1);
		diag_cov_inv_y.at<double>(i, i) = 1 / diag_cov_y.at<double>(i, i);
	}
	*/
/*
	std::cout << "diag_cov_x = " << diag_cov_x << std::endl;
	std::cout << "diag_cov_y = " << diag_cov_y << std::endl;
	return;
}
*/
Mat LevMarqBA::computedRdq(const Quaternion q)
{
	Mat dRdq(9, 4, CV_64F);

	Mat((Mat_<double>(1, 4) << 0, -4 * q.y, -4 * q.z, 0)).copyTo(dRdq.row(0));
	Mat((Mat_<double>(1, 4) << 2 * q.y, 2 * q.x, -2 * q.w, -2 * q.z)).copyTo(dRdq.row(1));
	Mat((Mat_<double>(1, 4) << 2 * q.z, 2 * q.w, 2 * q.x, 2 * q.y)).copyTo(dRdq.row(2));
	Mat((Mat_<double>(1, 4) << 2 * q.y, 2 * q.x, 2 * q.w, 2 * q.z)).copyTo(dRdq.row(3));
	Mat((Mat_<double>(1, 4) << -4 * q.x, 0, -4 * q.z, 0)).copyTo(dRdq.row(4));
	Mat((Mat_<double>(1, 4) << -2 * q.w, 2 * q.z, 2 * q.y, 2 * q.x)).copyTo(dRdq.row(5));
	Mat((Mat_<double>(1, 4) << 2 * q.z, -2 * q.w, 2 * q.x, -2 * q.y)).copyTo(dRdq.row(6));
	Mat((Mat_<double>(1, 4) << 2 * q.w, 2 * q.z, 2 * q.y, 2 * q.x)).copyTo(dRdq.row(7));
	Mat((Mat_<double>(1, 4) << -4 * q.x, -4 * q.y, 0, 0)).copyTo(dRdq.row(8));

	//std::cout << "dfdq = " << dRdq << std::endl;
	return dRdq;
}

Mat LevMarqBA::computedfdR(const Point3f xyz_point,
	const Point3f uvw_point,
	const Point3f C,
	const Mat& R,
	const CameraParams& camera)
{
	Mat du_dr = (Mat_<double>(1, 9) << camera.focal * (xyz_point.x - C.x),
		camera.focal * (xyz_point.y - C.y),
		camera.focal * (xyz_point.z - C.z),
		0.0, 0.0, 0.0,
		camera.ppx * (xyz_point.x - C.x),
		camera.ppx * (xyz_point.y - C.y),
		camera.ppx * (xyz_point.z - C.z));

	Mat dv_dr = (Mat_<double>(1, 9) << 0.0, 0.0, 0.0,
		camera.focal * (xyz_point.x - C.x),
		camera.focal * (xyz_point.y - C.y),
		camera.focal * (xyz_point.z - C.z),
		camera.ppy * (xyz_point.x - C.x),
		camera.ppy * (xyz_point.y - C.y),
		camera.ppy * (xyz_point.z - C.z));


	Mat dw_dr = (Mat_<double>(1, 9) << 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0,
		xyz_point.x - C.x,
		xyz_point.y - C.y,
		xyz_point.z - C.z);

	Mat dfdR(2, 9, CV_64F);

	Mat((uvw_point.z * du_dr - uvw_point.x * dw_dr) / (uvw_point.z * uvw_point.z)).row(0).copyTo(dfdR.row(0));
	Mat((uvw_point.z * dv_dr - uvw_point.y * dw_dr) / (uvw_point.z * uvw_point.z)).row(0).copyTo(dfdR.row(1));
	return dfdR;

}

Mat LevMarqBA::computedfdX(const Point3f xyz_point,
	const Point3f uvw_point,
	const Mat& R,
	const CameraParams& camera)
{
	Mat du_dx = (Mat_<double>(1, 3) << camera.focal * R.at<double>(0, 0) + camera.ppx * R.at<double>(2, 0),
		camera.focal * R.at<double>(0, 1) + camera.ppx * R.at<double>(2, 1),
		camera.focal * R.at<double>(0, 2) + camera.ppx * R.at<double>(2, 2));

	Mat dv_dx = (Mat_<double>(1, 3) << camera.focal * R.at<double>(1, 0) + camera.ppy * R.at<double>(2, 0),
		camera.focal * R.at<double>(1, 1) + camera.ppy * R.at<double>(2, 1),
		camera.focal * R.at<double>(2, 2) + camera.ppy * R.at<double>(2, 2));

	Mat dw_dx = (Mat_<double>(1, 3) << R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2));

	Mat dfdX(2, 3, CV_64F);

	Mat((uvw_point.z * du_dx - uvw_point.x * dw_dx) / (uvw_point.z * uvw_point.z)).copyTo(dfdX.row(0));
	Mat((uvw_point.z * dv_dx - uvw_point.y * dw_dx) / (uvw_point.z * uvw_point.z)).copyTo(dfdX.row(1));

	return dfdX;
}

Mat LevMarqBA::computedfdq(const Point3f xyz_point,
	const Point3f uvw_point,
	const Point3f C,
	const Mat& R,
	const CameraParams& camera)
{
	Mat dfdR = computedfdR(xyz_point, uvw_point, C, R, camera);
	Quaternion q = RMatToQuaternion(R);
	Mat dRdq = computedRdq(q);
	Mat dfdq = dfdR * dRdq;
	return dfdq;
}

Mat LevMarqBA::computedfdC(const Point3f xyz_point,
	const Point3f uvw_point,
	const Mat& R,
	const CameraParams& camera)
{
	Mat du_dx = (Mat_<double>(1, 3) << -(camera.focal * R.at<double>(0, 0) + camera.ppx * R.at<double>(2, 0)),
		-(camera.focal * R.at<double>(0, 1) + camera.ppx * R.at<double>(2, 1)),
		-(camera.focal * R.at<double>(0, 2) + camera.ppx * R.at<double>(2, 2)));

	Mat dv_dx = (Mat_<double>(1, 3) << -(camera.focal * R.at<double>(1, 0) + camera.ppy * R.at<double>(2, 0)),
		-(camera.focal * R.at<double>(1, 1) + camera.ppy * R.at<double>(2, 1)),
		-(camera.focal * R.at<double>(2, 2) + camera.ppy * R.at<double>(2, 2)));

	Mat dw_dx = (Mat_<double>(1, 3) << -R.at<double>(2, 0), -R.at<double>(2, 1), -R.at<double>(2, 2));

	Mat dfdC(2, 3, CV_64F);

	Mat((uvw_point.z * du_dx - uvw_point.x * dw_dx) / (uvw_point.z * uvw_point.z)).copyTo(dfdC.row(0));
	Mat((uvw_point.z * dv_dx - uvw_point.y * dw_dx) / (uvw_point.z * uvw_point.z)).copyTo(dfdC.row(1));

	return dfdC;
}

Quaternion LevMarqBA::RMatToQuaternion(const Mat R)
{
	std::vector<double> Q_vec(4);
	double trace = R.at<double>(0, 0) + R.at<double>(1, 1) + R.at<double>(2, 2);

	if (trace > 0.0)
	{
		double s = sqrt(trace + 1.0);
		Q_vec[3] = (s * 0.5);
		s = 0.5 / s;
		Q_vec[0] = ((R.at<double>(2, 1) - R.at<double>(1, 2)) * s);
		Q_vec[1] = ((R.at<double>(0, 2) - R.at<double>(2, 0)) * s);
		Q_vec[2] = ((R.at<double>(1, 0) - R.at<double>(0, 1)) * s);
	}
	else
	{
		int i = R.at<double>(0, 0) < R.at<double>(1, 1) ? (R.at<double>(1, 1) < R.at<double>(2, 2) ? 2 : 1) :
			(R.at<double>(0, 0) < R.at<double>(2, 2) ? 2 : 0);
		int j = (i + 1) % 3;
		int k = (i + 2) % 3;

		double s = sqrt(R.at<double>(i, i) - R.at<double>(j, j) - R.at<double>(k, k) + 1.0);
		Q_vec[i] = s * 0.5;
		s = 0.5 / s;

		Q_vec[3] = (R.at<double>(k, j) - R.at<double>(j, k)) * s;
		Q_vec[j] = (R.at<double>(j, i) + R.at<double>(i, j)) * s;
		Q_vec[k] = (R.at<double>(k, i) + R.at<double>(i, k)) * s;
	}
	//std::cout << Q_vec[0] << std::endl;
	return Quaternion(Q_vec);
}

Mat LevMarqBA::computeJacobian(const std::vector<Point3f>& X_3d,
	const std::vector<std::vector<int> >& visibility_indices,
	const std::vector<Mat>& R,
	const std::vector<Mat>& t,
	const CameraParams& camera)
{
	A = Mat::zeros(2 * n_views * n_points, 7 * n_views, CV_64F);
	B = Mat::zeros(2 * n_views * n_points, 3 * n_points, CV_64F);
	X_hat.resize(n_views);

	for (int i = 0; i < n_views; i++)
	{
		X_hat[i].resize(n_points);

		for (int j = 0; j < n_points; j++)
		{
			if (X[i][j] != Point2f(-1.0, -1.0))
			{
				Point3f C(Mat(-1 * R[i].inv() * t[i]));
				Point3f P = computeProjection(camera.getCameraParamsMat(), R[i], t[i], X_3d[j]);
				X_hat[i][j] = Point2f(P.x / P.z, P.y / P.z);

				Mat A_ij;
				hconcat(computedfdq(X_3d[j], P, C, R[i], camera),
					computedfdC(X_3d[j], P, R[i], camera),
					A_ij);
				A_ij.copyTo(A(Rect(7 * i, 2 * n_views * j, A_ij.cols, A_ij.rows)));

				Mat B_ij;
				B_ij = computedfdX(X_3d[j], P, R[i], camera);
				B_ij.copyTo(B(Rect(3 * j, 2 * (3 * j + i), B_ij.cols, B_ij.rows)));
			}
		}
	}
	hconcat(A, B, J);
	return J;
	/*
	Point3d C(Mat(-1 * R[0].inv() * t[0]));
	A_x.create(Size(7, X_3d.size()), CV_64F);
	A_y.create(Size(7, X_3d.size()), CV_64F);

	B_x.create(Size(X_3d.size() * 3, X_3d.size()), CV_64F);
	B_y.create(Size(X_3d.size() * 3, X_3d.size()), CV_64F);

	B_x = Mat::zeros(B_x.size(), B_x.type());
	B_y = Mat::zeros(B_y.size(), B_y.type());
	for (int i = 0; i < X.size(); i++)
	{
		Point3d P = computeProjection(camera.getCameraParamsMat(), R[0], t[0], X_3d[i]);
		X_hat[i] = Point2d(P.x / P.z, P.y / P.z);
		Mat A_i;
		hconcat(computedfdq(X_3d[i], P, C, R[0], camera),
				computedfdC(X_3d[i], P, R[0], camera),
				A_i);
		A_i.row(0).copyTo(A_x(Rect(0, i, A_i.cols, 1)));
		A_i.row(1).copyTo(A_y(Rect(0, i, A_i.cols, 1)));

		Mat B_i;
		B_i = computedfdX(X_3d[i], P, R[0], camera);
		B_i.row(0).copyTo(B_x(Rect(i * 3, i, B_i.cols, 1)));
		B_i.row(1).copyTo(B_y(Rect(i * 3, i, B_i.cols, 1)));
	}

	hconcat(A_x, B_x, J_x);
	hconcat(A_y, B_y, J_y);
	*/
	/*
	Mat J, A, B;
	A.create(Size(7, X.size() * 2), CV_64F);
	B.create(Size(X.size() * 3, X.size() * 2), CV_64F);
	B = Mat::zeros(B.size(), B.type());
	for (int i = 0; i < X.size(); i++)
	{
		Point3d P = computeProjection(camera[0], R[0], t[0], X[i]);
		Mat A_i;
		hconcat(computedfdq(X[i], P, C, R[0], camera[0], camera_params[0]),
				computedfdC(X[i], P, R[0], camera[0], camera_params[0]),
				A_i);
		A_i.copyTo(A(Rect(0, i * 2, A_i.cols, A_i.rows)));

		Mat B_i;
		B_i = computedfdX(X[i], P, R[0], camera[0], camera_params[0]);
		B_i.copyTo(B(Rect(i * 3, i * 2, B_i.cols, B_i.rows)));
	}

	hconcat(A, B, J);
	return J;
	*/
}

Mat LevMarqBA::augmentateNonZeroElements(const Mat& A, const double mu)
{
	std::vector<Point> non_zero_locations;
	findNonZero(A, non_zero_locations);
	Mat A_augm = A.clone();
	for (size_t i = 0; i < non_zero_locations.size(); i++)
	{
		A_augm.at<double>(non_zero_locations[i]) += mu;
	}
	return A_augm;
}

void LevMarqBA::findDelta(const double mu)
{
	Mat diag_cov = computeDiagCovarMatrix(X);
	Mat diag_cov_inv = diag_cov_inv.inv();

	U = A.t() * diag_cov_inv * A;
	V = B.t() * diag_cov_inv * B;
	W = A.t() * diag_cov_inv * B;

	U_augm = augmentateNonZeroElements(U, mu);
	V_augm = augmentateNonZeroElements(V, mu);

	Mat errs = computeErrors(X, X_hat);
	
	errors_a = A.t() * diag_cov_inv * errs;
	errors_b = B.t() * diag_cov_inv * errs;

	vconcat(errors_a, errors_b, errors);
	
	Y = W * V_augm.inv();

	Mat left_side = U_augm - (Y * W.t());
	Mat right_side = errors_a - (Y * errors_b);

	Mat delta_a;
	solve(left_side, right_side, delta_a, DECOMP_SVD);

	Mat delta_b;
	delta_b = Mat(V_augm.inv() * (errors_b - (W.t() * delta_a))).clone();

	std::cout << "delta_a" << delta_a << std::endl;
	std::cout << "delta_b" << delta_b << std::endl;
}