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

#ifndef __OPENCV_SFM_LIBMV_CAPI__
#define __OPENCV_SFM_LIBMV_CAPI__

#include "libmv/logging/logging.h"

#include "libmv/correspondence/feature.h"
#include "libmv/correspondence/feature_matching.h"
#include "libmv/correspondence/matches.h"
#include "libmv/correspondence/nRobustViewMatching.h"

#include "libmv/simple_pipeline/bundle.h"
#include "libmv/simple_pipeline/camera_intrinsics.h"
#include "libmv/simple_pipeline/keyframe_selection.h"
#include "libmv/simple_pipeline/initialize_reconstruction.h"
#include "libmv/simple_pipeline/pipeline.h"
#include "libmv/simple_pipeline/reconstruction_scale.h"
#include "libmv/simple_pipeline/tracks.h"
#include "gflags/gflags.h"

using namespace cv;
using namespace cv::sfm;
using namespace libmv;

using namespace google;

namespace gflags {}
using namespace gflags;

////////////////////////////////////////
// Based on 'libmv_capi' (blender API)
///////////////////////////////////////

struct libmv_Reconstruction {
  EuclideanReconstruction reconstruction;
  /* Used for per-track average error calculation after reconstruction */
  Tracks tracks;
  CameraIntrinsics *intrinsics;
  double error;
  bool is_valid;
};


//////////////////////////////////////
// Based on 'libmv_capi' (blender API)
/////////////////////////////////////

void libmv_initLogging(const char* argv0) {
  // Make it so FATAL messages are always print into console.
  char severity_fatal[32];
  static int initLog=0;
  snprintf(severity_fatal, sizeof(severity_fatal), "%d",
           GLOG_FATAL);

  if (!initLog)
      InitGoogleLogging(argv0);
  initLog=1;
  SetCommandLineOption("logtostderr", "1");
  SetCommandLineOption("v", "0");
  SetCommandLineOption("stderrthreshold", severity_fatal);
  SetCommandLineOption("minloglevel", severity_fatal);
}

void libmv_startDebugLogging(void) {
  SetCommandLineOption("logtostderr", "1");
  SetCommandLineOption("v", "2");
  SetCommandLineOption("stderrthreshold", "1");
  SetCommandLineOption("minloglevel", "0");
}

void libmv_setLoggingVerbosity(int verbosity) {
  char val[10];
  snprintf(val, sizeof(val), "%d", verbosity);
  SetCommandLineOption("v", val);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
// Based on the 'selectTwoKeyframesBasedOnGRICAndVariance()' function from 'libmv_capi' (blender API)
///////////////////////////////////////////////////////////////////////////////////////////////////////

/* Select the two keyframes that give a lower reprojection error
 */

bool selectTwoKeyframesBasedOnGRICAndVariance(
    Tracks& tracks,
    Tracks& normalized_tracks,
    CameraIntrinsics& camera_intrinsics,
    int& keyframe1,
    int& keyframe2) {

  libmv::vector<int> keyframes;

  /* Get list of all keyframe candidates first. */
  SelectKeyframesBasedOnGRICAndVariance(normalized_tracks,
                                        camera_intrinsics,
                                        keyframes);

  if (keyframes.size() < 2) {
    LG << "Not enough keyframes detected by GRIC";
    return false;
  } else if (keyframes.size() == 2) {
    keyframe1 = keyframes[0];
    keyframe2 = keyframes[1];
    return true;
  }

  /* Now choose two keyframes with minimal reprojection error after initial
   * reconstruction choose keyframes with the least reprojection error after
   * solving from two candidate keyframes.
   *
   * In fact, currently libmv returns single pair only, so this code will
   * not actually run. But in the future this could change, so let's stay
   * prepared.
   */
  int previous_keyframe = keyframes[0];
  double best_error = std::numeric_limits<double>::max();
  for (int i = 1; i < keyframes.size(); i++) {
    EuclideanReconstruction reconstruction;
    int current_keyframe = keyframes[i];
    libmv::vector<Marker> keyframe_markers =
      normalized_tracks.MarkersForTracksInBothImages(previous_keyframe,
                                                     current_keyframe);

    Tracks keyframe_tracks(keyframe_markers);

    /* get a solution from two keyframes only */
    EuclideanReconstructTwoFrames(keyframe_markers, &reconstruction);
    EuclideanBundle(keyframe_tracks, &reconstruction);
    EuclideanCompleteReconstruction(keyframe_tracks,
                                &reconstruction,
                                NULL);

    double current_error = EuclideanReprojectionError(tracks,
                                                      reconstruction,
                                                      camera_intrinsics);

    LG << "Error between " << previous_keyframe
       << " and " << current_keyframe
       << ": " << current_error;

    if (current_error < best_error) {
      best_error = current_error;
      keyframe1 = previous_keyframe;
      keyframe2 = current_keyframe;
    }

    previous_keyframe = current_keyframe;
  }

  return true;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// Based on the 'libmv_cameraIntrinsicsFillFromOptions()' function from 'libmv_capi' (blender API)
////////////////////////////////////////////////////////////////////////////////////////////////////

/* Fill the camera intrinsics parameters given the camera instrinsics
 * options values.
 */

static void libmv_cameraIntrinsicsFillFromOptions(
    const libmv_CameraIntrinsicsOptions* camera_intrinsics_options,
    CameraIntrinsics* camera_intrinsics) {
  camera_intrinsics->SetFocalLength(camera_intrinsics_options->focal_length,
                                    camera_intrinsics_options->focal_length);

  camera_intrinsics->SetPrincipalPoint(
      camera_intrinsics_options->principal_point_x,
      camera_intrinsics_options->principal_point_y);

  camera_intrinsics->SetImageSize(camera_intrinsics_options->image_width,
      camera_intrinsics_options->image_height);

  switch (camera_intrinsics_options->distortion_model) {
    case SFM_DISTORTION_MODEL_POLYNOMIAL:
      {
        PolynomialCameraIntrinsics *polynomial_intrinsics =
          static_cast<PolynomialCameraIntrinsics*>(camera_intrinsics);

        polynomial_intrinsics->SetRadialDistortion(
            camera_intrinsics_options->polynomial_k1,
            camera_intrinsics_options->polynomial_k2,
            camera_intrinsics_options->polynomial_k3);

        break;
      }

    case SFM_DISTORTION_MODEL_DIVISION:
      {
        DivisionCameraIntrinsics *division_intrinsics =
          static_cast<DivisionCameraIntrinsics*>(camera_intrinsics);

        division_intrinsics->SetDistortion(
            camera_intrinsics_options->division_k1,
            camera_intrinsics_options->division_k2);
        break;
      }

    default:
      assert(!"Unknown distortion model");
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////
// Based on the 'libmv_cameraIntrinsicsCreateFromOptions()' function from 'libmv_capi' (blender API)
//////////////////////////////////////////////////////////////////////////////////////////////////////

/* Create the camera intrinsics model given the camera instrinsics
 * options values.
 */

CameraIntrinsics* libmv_cameraIntrinsicsCreateFromOptions(
    const libmv_CameraIntrinsicsOptions* camera_intrinsics_options) {
  CameraIntrinsics *camera_intrinsics = NULL;
  switch (camera_intrinsics_options->distortion_model) {
    case SFM_DISTORTION_MODEL_POLYNOMIAL:
      camera_intrinsics = new PolynomialCameraIntrinsics();
      break;
    case SFM_DISTORTION_MODEL_DIVISION:
      camera_intrinsics = new DivisionCameraIntrinsics();
      break;
    default:
      assert(!"Unknown distortion model");
  }
  libmv_cameraIntrinsicsFillFromOptions(camera_intrinsics_options,
                                        camera_intrinsics);
  return camera_intrinsics;
}


////////////////////////////////////////////////////////////////////////////////////////
// Based on the 'libmv_getNormalizedTracks()' function from 'libmv_capi' (blender API)
////////////////////////////////////////////////////////////////////////////////////////

/* Normalizes the tracks given the camera intrinsics parameters
 */

void
libmv_getNormalizedTracks(const libmv::Tracks &tracks,
                          const libmv::CameraIntrinsics &camera_intrinsics,
                          libmv::Tracks *normalized_tracks) {
  libmv::vector<libmv::Marker> markers = tracks.AllMarkers();
  for (int i = 0; i < markers.size(); ++i) {
    libmv::Marker &marker = markers[i];
    camera_intrinsics.InvertIntrinsics(marker.x, marker.y,
                                       &marker.x, &marker.y);
    normalized_tracks->Insert(marker.image,
                              marker.track,
                              marker.x, marker.y,
                              marker.weight);
  }
}


//////////////////////////////////////////////////////////////////////////////////////////
// Based on the 'libmv_solveRefineIntrinsics()' function from 'libmv_capi' (blender API)
//////////////////////////////////////////////////////////////////////////////////////////

/* Refine the final solution using Bundle Adjustment
 */

void libmv_solveRefineIntrinsics(
    const Tracks &tracks,
    const int refine_intrinsics,
    const int bundle_constraints,
    EuclideanReconstruction* reconstruction,
    CameraIntrinsics* intrinsics) {
  /* only a few combinations are supported but trust the caller/ */
  int bundle_intrinsics = 0;

  if (refine_intrinsics & SFM_REFINE_FOCAL_LENGTH) {
    bundle_intrinsics |= libmv::BUNDLE_FOCAL_LENGTH;
  }
  if (refine_intrinsics & SFM_REFINE_PRINCIPAL_POINT) {
    bundle_intrinsics |= libmv::BUNDLE_PRINCIPAL_POINT;
  }
  if (refine_intrinsics & SFM_REFINE_RADIAL_DISTORTION_K1) {
    bundle_intrinsics |= libmv::BUNDLE_RADIAL_K1;
  }
  if (refine_intrinsics & SFM_REFINE_RADIAL_DISTORTION_K2) {
    bundle_intrinsics |= libmv::BUNDLE_RADIAL_K2;
  }

  EuclideanBundleCommonIntrinsics(tracks,
                                  bundle_intrinsics,
                                  bundle_constraints,
                                  reconstruction,
                                  intrinsics);
}


///////////////////////////////////////////////////////////////////////////////////
// Based on the 'finishReconstruction()' function from 'libmv_capi' (blender API)
///////////////////////////////////////////////////////////////////////////////////

/* Finish the reconstrunction and computes the final reprojection error
 */

void finishReconstruction(
    const Tracks &tracks,
    const CameraIntrinsics &camera_intrinsics,
    libmv_Reconstruction *libmv_reconstruction) {
  EuclideanReconstruction &reconstruction =
    libmv_reconstruction->reconstruction;

  /* Reprojection error calculation. */
  libmv_reconstruction->tracks = tracks;
  libmv_reconstruction->error = EuclideanReprojectionError(tracks,
                                                           reconstruction,
                                                           camera_intrinsics);
}


////////////////////////////////////////////////////////////////////////////////////////
// Based on the 'libmv_solveReconstruction()' function from 'libmv_capi' (blender API)
////////////////////////////////////////////////////////////////////////////////////////

/* Perform the complete reconstruction process
 */

libmv_Reconstruction *libmv_solveReconstruction(
    const Tracks &libmv_tracks,
    const libmv_CameraIntrinsicsOptions* libmv_camera_intrinsics_options,
    libmv_ReconstructionOptions* libmv_reconstruction_options) {
  libmv_Reconstruction *libmv_reconstruction =
    new libmv_Reconstruction();

  Tracks tracks = libmv_tracks;
  EuclideanReconstruction &reconstruction =
    libmv_reconstruction->reconstruction;

  /* Retrieve reconstruction options from C-API to libmv API. */
  CameraIntrinsics *camera_intrinsics;
  camera_intrinsics = libmv_reconstruction->intrinsics =
    libmv_cameraIntrinsicsCreateFromOptions(libmv_camera_intrinsics_options);

  /* Invert the camera intrinsics/ */
  Tracks normalized_tracks;
  libmv_getNormalizedTracks(tracks, *camera_intrinsics, &normalized_tracks);

  /* keyframe selection. */
  int keyframe1 = libmv_reconstruction_options->keyframe1,
      keyframe2 = libmv_reconstruction_options->keyframe2;

  if (libmv_reconstruction_options->select_keyframes) {
    LG << "Using automatic keyframe selection";

    selectTwoKeyframesBasedOnGRICAndVariance(tracks,
                                             normalized_tracks,
                                             *camera_intrinsics,
                                             keyframe1,
                                             keyframe2);

    /* so keyframes in the interface would be updated */
    libmv_reconstruction_options->keyframe1 = keyframe1;
    libmv_reconstruction_options->keyframe2 = keyframe2;
  }

  /* Actual reconstruction. */
  LG << "frames to init from: " << keyframe1 << " " << keyframe2;

  libmv::vector<Marker> keyframe_markers =
    normalized_tracks.MarkersForTracksInBothImages(keyframe1, keyframe2);

  LG << "number of markers for init: " << keyframe_markers.size();

  if (keyframe_markers.size() < 8) {
    LG << "No enough markers to initialize from";
    libmv_reconstruction->is_valid = false;
    return libmv_reconstruction;
  }

  EuclideanReconstructTwoFrames(keyframe_markers, &reconstruction);
  EuclideanBundle(normalized_tracks, &reconstruction);
  EuclideanCompleteReconstruction(normalized_tracks,
                                  &reconstruction,
                                  NULL);

  /* Refinement/ */
  if (libmv_reconstruction_options->refine_intrinsics) {
    libmv_solveRefineIntrinsics(
                                tracks,
                                libmv_reconstruction_options->refine_intrinsics,
                                libmv::BUNDLE_NO_CONSTRAINTS,
                                &reconstruction,
                                camera_intrinsics);
  }

  /* Set reconstruction scale to unity. */
  EuclideanScaleToUnity(&reconstruction);

  finishReconstruction(tracks,
                       *camera_intrinsics,
                       libmv_reconstruction);

  libmv_reconstruction->is_valid = true;
  return (libmv_Reconstruction *) libmv_reconstruction;
}

#endif
