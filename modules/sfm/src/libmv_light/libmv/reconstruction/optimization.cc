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

#include "libmv/multiview/bundle.h"
#include "libmv/reconstruction/optimization.h"
#include "libmv/reconstruction/tools.h"

namespace libmv {

double EstimateRootMeanSquareError(const Matches &matches,
                                   Reconstruction *reconstruction) {
  PinholeCamera * pcamera = NULL;
  vector<StructureID> structures_ids;
  Mat2X x_image;
  Mat4X X_world;
  double sum_rms2 = 0;
  size_t num_features = 0;
  std::map<StructureID, Structure *>::iterator stucture_iter;
  std::map<CameraID, Camera *>::iterator camera_iter =
      reconstruction->cameras().begin();
  for (; camera_iter != reconstruction->cameras().end(); ++camera_iter) {
    pcamera = dynamic_cast<PinholeCamera *>(camera_iter->second);
    if (pcamera) {
      SelectExistingPointStructures(matches, camera_iter->first,
                                    *reconstruction,
                                    &structures_ids,
                                    &x_image);
      MatrixOfPointStructureCoordinates(structures_ids,
                                        *reconstruction,
                                        &X_world);
      Mat2X dx =Project(pcamera->projection_matrix(), X_world) - x_image;
      VLOG(1)   << "|Err Cam "<<camera_iter->first<<"| = "
                << sqrt(Square(dx.norm()) / x_image.cols()) << " ("
                << x_image.cols() << " pts)" << std::endl;
      // TODO(julien) use normSquare
      sum_rms2 += Square(dx.norm());
      num_features += x_image.cols();
    }
  }
  // TODO(julien) devide by total number of features
  return sqrt(sum_rms2 / num_features);
}

double MetricBundleAdjust(const Matches &matches,
                          Reconstruction *reconstruction) {
  double rms = 0, rms0 = EstimateRootMeanSquareError(matches, reconstruction);
  VLOG(1)   << "Initial RMS = " << rms0 << std::endl;
  size_t ncamera = reconstruction->GetNumberCameras();
  size_t nstructure = reconstruction->GetNumberStructures();
  vector<Mat2X> x(ncamera);
  vector<Vecu>   x_ids(ncamera);
  vector<Mat3>  Ks(ncamera);
  vector<Mat3>  Rs(ncamera);
  vector<Vec3>  ts(ncamera);
  Mat3X         X(3, nstructure);
  vector<StructureID> structures_ids;
  std::map<StructureID, uint> map_structures_ids;

  size_t str_id = 0;
  PointStructure *pstructure = NULL;
  std::map<StructureID, Structure *>::iterator str_iter =
    reconstruction->structures().begin();
  for (; str_iter != reconstruction->structures().end(); ++str_iter) {
    pstructure = dynamic_cast<PointStructure *>(str_iter->second);
    if (pstructure) {
      X.col(str_id) = pstructure->coords_affine();
      map_structures_ids[str_iter->first] = str_id;
      str_id++;
    } else {
      LOG(FATAL) << "Error: the bundle adjustment cannot handle non point "
                 << "structure.";
      return 0;
    }
  }

  PinholeCamera * pcamera = NULL;
  size_t cam_id = 0;
  std::map<CameraID, Camera *>::iterator cam_iter =
    reconstruction->cameras().begin();
  for (; cam_iter != reconstruction->cameras().end(); ++cam_iter) {
    pcamera = dynamic_cast<PinholeCamera *>(cam_iter->second);
    if (pcamera) {
      pcamera->GetIntrinsicExtrinsicParameters(&Ks[cam_id],
                                               &Rs[cam_id],
                                               &ts[cam_id]);
      SelectExistingPointStructures(matches, cam_iter->first,
                                    *reconstruction,
                                    &structures_ids, &x[cam_id]);
      x_ids[cam_id].resize(structures_ids.size());
      for (size_t s = 0; s < structures_ids.size(); ++s) {
        x_ids[cam_id][s] = map_structures_ids[structures_ids[s]];
      }
      //VLOG(1)   << "x_ids = " << x_ids[cam_id].transpose()<<"\n";
      cam_id++;
    } else {
      LOG(FATAL) << "Error: the bundle adjustment cannot handle non pinhole "
                 << "cameras.";
      return 0;
    }
  }
  // Performs metric bundle adjustment
  rms = EuclideanBA(x, x_ids, &Ks, &Rs, &ts, &X, eBUNDLE_METRIC);
  // Copy the results only if it's better
  if (rms < rms0) {
    cam_id = 0;
    cam_iter = reconstruction->cameras().begin();
    for (; cam_iter != reconstruction->cameras().end(); ++cam_iter) {
      pcamera = dynamic_cast<PinholeCamera *>(cam_iter->second);
      if (pcamera) {
        pcamera->SetIntrinsicExtrinsicParameters(Ks[cam_id],
                                                 Rs[cam_id],
                                                 ts[cam_id]);
        cam_id++;
      }
    }
    str_id = 0;
    str_iter = reconstruction->structures().begin();
    for (; str_iter != reconstruction->structures().end(); ++str_iter) {
      pstructure = dynamic_cast<PointStructure *>(str_iter->second);
      if (pstructure) {
        pstructure->set_coords_affine(X.col(str_id));
        str_id++;
      }
    }
  }
  //rms = EstimateRootMeanSquareError(matches, reconstruction);
  VLOG(1)   << "Final RMS = " << rms << std::endl;
  return rms;
}

uint RemoveOutliers(CameraID image_id,
                    Matches *matches,
                    Reconstruction *reconstruction,
                    double rmse_threshold) {
  // TODO(julien) finish it !
  // Checks that the camera is in reconstruction
  if (!reconstruction->ImageHasCamera(image_id)) {
    return 0;
  }
  vector<StructureID> structures_ids;
  // Selects only the reconstructed structures observed in the image
  SelectExistingPointStructures(*matches, image_id, *reconstruction,
                                &structures_ids, NULL);
  Vec2 q, q2;
  uint number_outliers = 0;
  uint num_views = 0;
  bool current_point_removed = false;
  PinholeCamera *camera = NULL;
  PointStructure *pstructure = NULL;
  double err = 0;
  for (size_t t = 0; t < structures_ids.size(); ++t) {
    pstructure = dynamic_cast<PointStructure *>(
      reconstruction->GetStructure(structures_ids[t]));
    if (pstructure) {
      Matches::Features<PointFeature> fp =
       matches->InTrack<PointFeature>(structures_ids[t]);
      current_point_removed = false;
      while (fp) {
        camera = dynamic_cast<PinholeCamera *>(
          reconstruction->GetCamera(fp.image()));
        if (camera) {
          q << fp.feature()->x(), fp.feature()->y();
          camera->ProjectPointStructure(*pstructure, &q2);
          err = (q - q2).norm();
          if (err > rmse_threshold) {
            matches->Remove(fp.image(), structures_ids[t]);
            if (!current_point_removed)
              number_outliers++;
            current_point_removed = true;
          }
        }
        fp.operator++();
      }
      // TODO(julien) put the check into a function
      // Check if a point has enough views (with pinhole cameras)
      fp = matches->InTrack<PointFeature>(structures_ids[t]);
      num_views = 0;
      while (fp) {
        camera = dynamic_cast<PinholeCamera *>(
          reconstruction->GetCamera(fp.image()));
        if (camera) {
          num_views++;
        }
        fp.operator++();
      }
      if (num_views < 2) {
        //Delete the point
        reconstruction->RemoveTrack(structures_ids[t]);
      }
      // TODO(julien) also check if a camera has enough points (at least 2)
    }
  }
  VLOG(1) << "#outliers: " << number_outliers << std::endl;
  return number_outliers;
}
} // namespace libmv
