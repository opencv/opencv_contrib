#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <iostream>
#include <sstream>

using namespace std;
using namespace cv;

void
generateScene(const size_t n_views, const size_t n_points, const bool is_projective, Matx33d & K, vector<Matx33d> & R,
              vector<Vec3d> & t, vector<Matx34d> & P, Mat_<double> & points3d,
              vector<Mat_<double> > & points2d );

void
parser_2D_tracks( const vector<Mat_<double> > &points2d, libmv::Tracks &tracks );

int main(int argc, char* argv[])
{
  float err_max2d = 1e-7;
  int nviews = 20;
  int npoints = 500;
  bool is_projective = true;
  bool has_outliers = false;
  bool is_sequence = true;

  // read input parameters
  if ( argc > 1)
  {
    nviews = atoi(argv[1]);

    if ( argc > 2 )
    {
        npoints = atoi(argv[2]);
    }
  }

  vector< Mat_<double> > points2d;
  vector< Matx33d > Rs;
  vector< Vec3d > ts;
  vector< Matx34d > Ps;
  Matx33d K;
  Mat_<double> points3d;

  /// Generate ground truth scene
  cout << "Generating scene" << endl;
  generateScene(nviews, npoints, is_projective, K, Rs, ts, Ps, points3d,
    points2d);

  cout << "Reconstructing scene" << endl;

  /// Reconstruct the scene using the 2d correspondences
  Mat_<double> points3d_estimated;
  vector<Mat> Rs_est;
  vector<Mat> ts_est;
  //reconstruct(points2d, Rs_est, ts_est, K, points3d_estimated, is_projective, has_outliers, is_sequence);

  vector< Mat > Ps_est;
  Matx33d K_est;
  vector< Mat_<double> > points2d_;
  points2d_.push_back(points2d[0]);
  points2d_.push_back(points2d[1]);
  reconstruct(points2d, Rs_est, ts_est, K, points3d_estimated, is_projective, has_outliers, is_sequence);


  /// Create 3D windows
  viz::Viz3d window_gt("Ground Truth Coordinate Frame");
  viz::Viz3d window_est("Estimation Coordinate Frame");

  /// Add coordinate axes
  window_gt.showWidget("Ground Truth Coordinate Widget", viz::WCoordinateSystem());
  window_est.showWidget("Estimation Coordinate Widget", viz::WCoordinateSystem());

  // Create the pointcloud
  cout << "Recovering points" << endl;

  vector<Vec3f> point_cloud;
  for (int i = 0; i < points3d.cols; ++i) {
    // recover ground truth points3d
    Vec3f point3d((float) points3d(0, i),
                  (float) points3d(1, i),
                  (float) points3d(2, i));
    point_cloud.push_back(point3d);
  }

  vector<Vec3f> point_cloud_est;
  for (int i = 0; i < points3d_estimated.cols; ++i) {

    // recover estimated points3d
    Vec3f point3d_est((float) points3d_estimated(0, i),
                      (float) points3d_estimated(1, i),
                      (float) points3d_estimated(2, i));
    point_cloud_est.push_back(point3d_est);
  }

  /// Add the pointcloud
  if ( !point_cloud.empty() && !point_cloud_est.empty() )
  {
    cout << "Rendering points" << endl;
    viz::WCloud cloud_widget(point_cloud, viz::Color::green());
    viz::WCloud cloud_est_widget(point_cloud_est, viz::Color::red());
    window_gt.showWidget("point_cloud", cloud_widget);
    window_est.showWidget("point_cloud_est", cloud_est_widget);
  }
  else
  {
    cout << "Cannot rendering points: empty pointcloud" << endl;
  }

  /// Add cameras
  cout << "Rendering Cameras" << endl;
  std::vector<Affine3d> path_gt;
  for (int i = 0, j = 1; i < nviews; ++i, ++j)
    path_gt.push_back(Affine3d(Rs[i],ts[i]));
  path_gt.push_back(Affine3d(Rs[0],ts[0]));

  std::vector<Affine3d> path_est;
  for (int i = 0, j = 1; i < nviews; ++i, ++j)
    path_est.push_back(Affine3d(Rs_est[i],ts_est[i]));
  path_est.push_back(Affine3d(Rs_est[0],ts_est[0]));

  window_gt.showWidget("cameras_frames_and_lines_gt", viz::WTrajectory(path_gt, viz::WTrajectory::BOTH, 0.2, viz::Color::green()));
  window_gt.showWidget("cameras_frustums_gt", viz::WTrajectoryFrustums(path_gt, K, 0.3, viz::Color::yellow()));
  window_est.showWidget("cameras_frames_and_lines_est", viz::WTrajectory(path_est, viz::WTrajectory::BOTH, 0.2, viz::Color::green()));
  window_est.showWidget("cameras_frustums_est", viz::WTrajectoryFrustums(path_est, K, 0.3, viz::Color::yellow()));
  window_gt.spin();
  window_est.spin();

  return 0;
}


void
generateScene(const size_t n_views, const size_t n_points, const bool is_projective, Matx33d & K, vector<Matx33d> & R,
              vector<Vec3d> & t, vector<Matx34d> & P, Mat_<double> & points3d,
              vector<Mat_<double> > & points2d)
{
  R.resize(n_views);
  t.resize(n_views);

  cv::RNG rng;

  // Generate a bunch of random 3d points in a 0, 1 cube
  points3d.create(3, n_points);
  rng.fill(points3d, cv::RNG::UNIFORM, 0, 10);

  // Generate random intrinsics
  K = Matx33d(500,   0, 320,
                0, 500, 240,
                0,   0,   1);

  float r = 10.0f;
  float cx = r/2.0f;
  float cy = r/2.0f;
  float cz = r/2.0f;
  int num_segments = n_views;

  for(int ii = 0; ii < num_segments; ii++)
  {
    float theta = 2.0f * CV_PI * float(ii) / float(num_segments);//get the current angle

    float x = r * cosf(theta);//calculate the x component
    float y = r * sinf(theta);//calculate the y component

    // set camera position
    t[ii] = cv::Vec3d(x + cx, y + cy, cz);//output vertex

    // set rotation around x and y axis
    Vec3d vecx(-CV_PI/2, 0, 0);
    Vec3d vecy(0, -CV_PI/2-theta, 0);
    Vec3d vecz(0, 0, 0);

    Matx33d Rx, Ry, Rz;
    Rodrigues(vecx, Rx);
    Rodrigues(vecy, Ry);
    Rodrigues(vecz, Rz);

    // apply ordered rotations
    R[ii] = Rx * Ry * Rz;
  }

  // Compute projection matrices
  P.resize(n_views);
  for (size_t i = 0; i < n_views; ++i)
  {
    Matx33d K3 = K, R3 = R[i];
    Vec3d t3 = t[i];
    P_From_KRt(K3, R3, t3, P[i]);
    //cout << P[i] << endl;
  }

  // Compute homogeneous 3d points
  Mat_<double> points3d_homogeneous(4, n_points);
  points3d.copyTo(points3d_homogeneous.rowRange(0, 3));
  points3d_homogeneous.row(3).setTo(1);

  // Project those points for every view
  points2d.resize(n_views);
  for (size_t i = 0; i < n_views; ++i)
  {
    Mat_<double> points2d_tmp = Mat(P[i]) * points3d_homogeneous;
    points2d[i].create(2, n_points);
    for (unsigned char j = 0; j < 2; ++j)
      Mat(points2d_tmp.row(j) / points2d_tmp.row(2)).copyTo(points2d[i].row(j));
  }

// TODO: remove a certain number of points per view
// TODO: add a certain number of outliers per view

}