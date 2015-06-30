// Copyright (c) 2010 libmv authors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include "libmv/base/vector_utils.h"
#include "libmv/camera/pinhole_camera.h"
#include "libmv/multiview/autocalibration.h"
#include "libmv/multiview/fundamental.h"
#include "libmv/multiview/robust_fundamental.h"
#include "libmv/multiview/robust_resection.h"
#include "libmv/reconstruction/mapping.h"
#include "libmv/reconstruction/optimization.h"
#include "libmv/reconstruction/tools.h"
#include "libmv/reconstruction/projective_reconstruction.h"

namespace libmv {

bool ReconstructFromTwoUncalibratedViews(const Matches &matches,
                                         CameraID image_id1,
                                         CameraID image_id2,
                                         Matches *matches_inliers,
                                         Reconstruction *reconstruction) {
  double epipolar_threshold = 1;// in pixels
  vector<Mat> xs(2);
  vector<Matches::TrackID> tracks;
  vector<Matches::ImageID> images;
  images.push_back(image_id1);
  images.push_back(image_id2);
  PointMatchMatrices(matches, images, &tracks, &xs);
  // TODO(julien) Also remove structures that are on the same location
  if (xs[0].cols() < 7) {
    LOG(ERROR) << "Error: there are not enough common matches ("
               << xs[0].cols()<< "<7).";
    return false;
  }

  Mat &x0 = xs[0];
  Mat &x1 = xs[1];
  vector<int> feature_inliers;
  Mat3 F;
  // Computes fundamental matrix
  // TODO(julien) For the calibrated case, we can squeeze the fundamental using
  // directly the 5 points algorithm
  FundamentalFromCorrespondences7PointRobust(x0, x1,
                                             epipolar_threshold,
                                             &F, &feature_inliers,
                                             1e-3);
  Mat34 P1;
  Mat34 P2;
  P1<< Mat3::Identity(), Vec3::Zero();
  PinholeCamera * pcamera = NULL;
  pcamera = dynamic_cast<PinholeCamera *>(
    reconstruction->GetCamera(image_id1));
  // If the first image has no associated camera, we choose the center of the
  // coordinate frame
  if (!pcamera) {
    pcamera = new PinholeCamera(P1);
    reconstruction->InsertCamera(image_id1, pcamera);
    VLOG(1)   << "Add Camera ["
              << image_id1 <<"]"<< std::endl <<"P="
              << P1 << std::endl;
  } else {
    // TODO(julien) what should we do?
    // for now we enforce the first projection matrix to be the world reference
    VLOG(1) << "Warning: the first projection matrix is overwritten to be the"
            << " world reference frame.";
    pcamera->set_projection_matrix(P1);
  }
  // Recover the second projection matrix
  ProjectionsFromFundamental(F, &P1, &P2);
  // Creates and adds the second camera
  pcamera = new PinholeCamera(P2);
  reconstruction->InsertCamera(image_id2, pcamera);
  VLOG(1)   << "Add Camera ["
            << image_id2 <<"]"<< std::endl <<"P="
              << P2 << std::endl;

  //Adds only inliers matches into
  const Feature * feature = NULL;
  for (size_t s = 0; s < feature_inliers.size(); ++s) {
    feature = matches.Get(image_id1, tracks[feature_inliers[s]]);
    matches_inliers->Insert(image_id1, tracks[feature_inliers[s]], feature);
    feature = matches.Get(image_id2, tracks[feature_inliers[s]]);
    matches_inliers->Insert(image_id2, tracks[feature_inliers[s]], feature);
  }
  VLOG(1)   << "Inliers added: " << feature_inliers.size() << std::endl;
  return true;
}

bool UncalibratedCameraResection(const Matches &matches,
                                 CameraID image_id,
                                 Matches *matches_inliers,
                                 Reconstruction *reconstruction) {
  double rms_inliers_threshold = 1;// in pixels
  vector<StructureID> structures_ids;
  Mat2X x_image;
  Mat4X X_world;
  // Selects only the reconstructed tracks observed in the image
  SelectExistingPointStructures(matches, image_id, *reconstruction,
                                &structures_ids, &x_image);

  // TODO(julien) Also remove structures that are on the same location
  if (structures_ids.size() < 6) {
    LOG(ERROR) << "Error: there are not enough points to estimate the "
               << "projection matrix(" << structures_ids.size() << "<6).";
    // We need at least 6 tracks in order to do resection
    return false;
  }

  MatrixOfPointStructureCoordinates(structures_ids, *reconstruction, &X_world);
  CHECK(x_image.cols() == X_world.cols());

  Mat34 P;
  vector<int> inliers;
  ResectionRobust(x_image, X_world, rms_inliers_threshold, &P,&inliers, 1e-3);

  // TODO(julien) Performs non-linear optimization of the pose.

  // Creates a new camera and add it to the reconstruction
  PinholeCamera * camera = new PinholeCamera(P);
  reconstruction->InsertCamera(image_id, camera);

  VLOG(1)   << "Add Camera ["
            << image_id <<"]"<< std::endl <<"P="
            << P << std::endl;
  //Adds only inliers matches into
  const Feature * feature = NULL;
  for (size_t s = 0; s < structures_ids.size(); ++s) {
    feature = matches.Get(image_id, structures_ids[s]);
    matches_inliers->Insert(image_id, structures_ids[s], feature);
  }
  VLOG(1)   << "Inliers added: " << structures_ids.size() << std::endl;
  return true;
}

bool UpgradeToMetric(const Matches &matches,
                     Reconstruction *reconstruction) {
  double rms = EstimateRootMeanSquareError(matches, reconstruction);
  VLOG(1)   << "Upgrade to Metric - Initial RMS:" << rms << std::endl;
  AutoCalibrationLinear auto_calibration_linear;
  uint image_width = 0;
  uint image_height = 0;
  PinholeCamera * pcamera = NULL;
  std::map<CameraID, Camera *>::iterator camera_iter =
    reconstruction->cameras().begin();
  for (; camera_iter != reconstruction->cameras().end(); ++camera_iter) {
    pcamera = dynamic_cast<PinholeCamera *>(camera_iter->second);
    if (pcamera) {
      image_width = pcamera->image_width();
      image_height = pcamera->image_height();
      // Avoid to have null values of image width and height
      // TODO(julien) prefer an assert?
      if (!image_width  && !image_height) {
        image_width  = 640;
        image_height = 480;
      }
      auto_calibration_linear.AddProjection(pcamera->projection_matrix(),
                                            image_width, image_height);
    }
  }
  // TODO(julien) Put the following in a function.
  // Upgrade the reconstruction to metric using {Pm, Xm} = {P*H, H^{-1}*X}
  Mat4 H = auto_calibration_linear.MetricTransformation();
  VLOG(1)   << "Rectification H = " << H << "\n";
  if (isnan(H.sum())) {
    LOG(ERROR) << "Warning: The metric rectification cannot be applied, the "
               << "matrix contains NaN values.\n";
    return false;
  }
  Mat34 P;
  camera_iter = reconstruction->cameras().begin();
  for (; camera_iter != reconstruction->cameras().end(); ++camera_iter) {
    pcamera = dynamic_cast<PinholeCamera *>(camera_iter->second);
    if (pcamera) {
      P = pcamera->projection_matrix() * H;
      pcamera->set_projection_matrix(P);
      P_From_KRt(pcamera->intrinsic_matrix(),
                 pcamera->orientation_matrix(),
                 pcamera->position(), &P);
      // TODO(julien) change this.
      pcamera->set_projection_matrix(P);
    }
  }
  Mat4 H_inverse = H.inverse();
  PointStructure * pstructure = NULL;
  std::map<StructureID, Structure *>::iterator stucture_iter =
    reconstruction->structures().begin();
  for (; stucture_iter != reconstruction->structures().end(); ++stucture_iter) {
    pstructure = dynamic_cast<PointStructure *>(stucture_iter->second);
    if (pstructure) {
      pstructure->set_coords(H_inverse * pstructure->coords());
    }
  }
  MetricBundleAdjust(matches, reconstruction);
  return true;
}
} // namespace libmv
