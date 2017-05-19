/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "precomp.hpp"

#if CERES_FOUND

#include <opencv2/core/eigen.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "libmv_capi.h"

using namespace std;

namespace cv
{
namespace sfm
{

/* Parses a given array of 2d points into the libmv tracks structure
 */

void
parser_2D_tracks( const std::vector<Mat> &points2d, libmv::Tracks &tracks )
{
  const int nframes = static_cast<int>(points2d.size());
  for (int frame = 0; frame < nframes; ++frame) {
    const int ntracks = points2d[frame].cols;
    for (int track = 0; track < ntracks; ++track) {
      const Vec2d track_pt = points2d[frame].col(track);
      if ( track_pt[0] > 0 && track_pt[1] > 0 )
        tracks.Insert(frame, track, track_pt[0], track_pt[1]);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////

/* Parses a given set of matches into the libmv tracks structure
 */

void
parser_2D_tracks( const libmv::Matches &matches, libmv::Tracks &tracks )
{
  std::set<Matches::ImageID>::const_iterator iter_image =
    matches.get_images().begin();

  bool is_first_time = true;

  for (; iter_image != matches.get_images().end(); ++iter_image) {
    // Exports points
    Matches::Features<PointFeature> pfeatures =
      matches.InImage<PointFeature>(*iter_image);

    while(pfeatures) {

      double x = pfeatures.feature()->x(),
             y = pfeatures.feature()->y();

      // valid marker
      if ( x > 0 && y > 0 )
      {
        tracks.Insert(*iter_image, pfeatures.track(), x, y);

        if ( is_first_time )
          is_first_time = false;
      }

      // lost track
      else if ( x < 0 && y < 0 )
      {
        is_first_time = true;
      }

      pfeatures.operator++();
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////

/* Computes the 2d features matches between a given set of images and call the
 * reconstruction pipeline.
 */

libmv_Reconstruction *libmv_solveReconstructionImpl(
  const std::vector<String> &images,
  const libmv_CameraIntrinsicsOptions* libmv_camera_intrinsics_options,
  libmv_ReconstructionOptions* libmv_reconstruction_options)
{
  Ptr<Feature2D> edetector = ORB::create(10000);
  Ptr<Feature2D> edescriber = xfeatures2d::DAISY::create();
  //Ptr<Feature2D> edescriber = xfeatures2d::LATCH::create(64, true, 4);
  std::vector<std::string> sImages;
  for (int i=0;i<images.size();i++)
      sImages.push_back(images[i].c_str());
  cout << "Initialize nViewMatcher ... ";
  libmv::correspondence::nRobustViewMatching nViewMatcher(edetector, edescriber);

  cout << "OK" << endl << "Performing Cross Matching ... ";
  nViewMatcher.computeCrossMatch(sImages); cout << "OK" << endl;

  // Building tracks
  libmv::Tracks tracks;
  libmv::Matches matches = nViewMatcher.getMatches();
  parser_2D_tracks( matches, tracks );

  // Perform reconstruction
  return libmv_solveReconstruction(tracks,
                                   libmv_camera_intrinsics_options,
                                   libmv_reconstruction_options);
}

///////////////////////////////////////////////////////////////////////////////////////////////

template <class T>
class SFMLibmvReconstructionImpl : public T
{
public:
  SFMLibmvReconstructionImpl(const libmv_CameraIntrinsicsOptions &camera_instrinsic_options,
                             const libmv_ReconstructionOptions &reconstruction_options) :
    libmv_reconstruction_options_(reconstruction_options),
    libmv_camera_intrinsics_options_(camera_instrinsic_options) {}

  /* Run the pipeline given 2d points
   */

  virtual void run(InputArrayOfArrays _points2d)
  {
    std::vector<Mat> points2d;
    _points2d.getMatVector(points2d);
    CV_Assert( _points2d.total() >= 2 );

    // Parse 2d points to Tracks
    Tracks tracks;
    parser_2D_tracks(points2d, tracks);

    // Set libmv logs level
    libmv_initLogging("");

    if (libmv_reconstruction_options_.verbosity_level >= 0)
    {
      libmv_startDebugLogging();
      libmv_setLoggingVerbosity(
        libmv_reconstruction_options_.verbosity_level);
    }

    // Perform reconstruction
    libmv_reconstruction_ =
      *libmv_solveReconstruction(tracks,
                                 &libmv_camera_intrinsics_options_,
                                 &libmv_reconstruction_options_);
  }

  virtual void run(InputArrayOfArrays points2d, InputOutputArray K, OutputArray Rs,
                   OutputArray Ts, OutputArray points3d)
  {
    // Run the pipeline
    run(points2d);

    // Extract Data
    extractLibmvReconstructionData(K, Rs, Ts, points3d);
  }


  /* Run the pipeline given a set of images
   */

  virtual void run(const std::vector <String> &images)
  {
    // Set libmv logs level
    libmv_initLogging("");

    if (libmv_reconstruction_options_.verbosity_level >= 0)
    {
      libmv_startDebugLogging();
      libmv_setLoggingVerbosity(
        libmv_reconstruction_options_.verbosity_level);
    }

    // Perform reconstruction

    libmv_reconstruction_ =
      *libmv_solveReconstructionImpl(images,
                                     &libmv_camera_intrinsics_options_,
                                     &libmv_reconstruction_options_);
  }


  virtual void run(const std::vector <String> &images, InputOutputArray K, OutputArray Rs,
                   OutputArray Ts, OutputArray points3d)
  {
    // Run the pipeline
    run(images);

    // Extract Data
    extractLibmvReconstructionData(K, Rs, Ts, points3d);
  }

  virtual double getError() const { return libmv_reconstruction_.error; }

  virtual void
  getPoints(OutputArray points3d) {
    const size_t n_points =
      libmv_reconstruction_.reconstruction.AllPoints().size();

    points3d.create(n_points, 1, CV_64F);

    Vec3d point3d;
    for ( size_t i = 0; i < n_points; ++i )
    {
      for ( int j = 0; j < 3; ++j )
        point3d[j] =
          libmv_reconstruction_.reconstruction.AllPoints()[i].X[j];
      Mat(point3d).copyTo(points3d.getMatRef(i));
    }

  }

  virtual cv::Mat getIntrinsics() const {
    Mat K;
    eigen2cv(libmv_reconstruction_.intrinsics->K(), K);
    return K;
  }

  virtual void
  getCameras(OutputArray Rs, OutputArray Ts) {
    const size_t n_views =
      libmv_reconstruction_.reconstruction.AllCameras().size();

    Rs.create(n_views, 1, CV_64F);
    Ts.create(n_views, 1, CV_64F);

    Matx33d R;
    Vec3d t;
    for(size_t i = 0; i < n_views; ++i)
    {
      eigen2cv(libmv_reconstruction_.reconstruction.AllCameras()[i].R, R);
      eigen2cv(libmv_reconstruction_.reconstruction.AllCameras()[i].t, t);
      Mat(R).copyTo(Rs.getMatRef(i));
      Mat(t).copyTo(Ts.getMatRef(i));
    }
  }

  virtual void setReconstructionOptions(
    const libmv_ReconstructionOptions &libmv_reconstruction_options) {
      libmv_reconstruction_options_ = libmv_reconstruction_options;
  }

  virtual void setCameraIntrinsicOptions(
    const libmv_CameraIntrinsicsOptions &libmv_camera_intrinsics_options) {
      libmv_camera_intrinsics_options_ = libmv_camera_intrinsics_options;
  }

private:

  void
  extractLibmvReconstructionData(InputOutputArray K,
                                 OutputArray Rs,
                                 OutputArray Ts,
                                 OutputArray points3d)
  {
    getCameras(Rs, Ts);
    getPoints(points3d);
    getIntrinsics().copyTo(K.getMat());
  }

  libmv_Reconstruction libmv_reconstruction_;
  libmv_ReconstructionOptions libmv_reconstruction_options_;
  libmv_CameraIntrinsicsOptions libmv_camera_intrinsics_options_;
};

///////////////////////////////////////////////////////////////////////////////////////////////

Ptr<SFMLibmvEuclideanReconstruction>
SFMLibmvEuclideanReconstruction::create(const libmv_CameraIntrinsicsOptions &camera_instrinsic_options,
                                        const libmv_ReconstructionOptions &reconstruction_options)
{
  return makePtr<SFMLibmvReconstructionImpl<SFMLibmvEuclideanReconstruction> >(camera_instrinsic_options,reconstruction_options);
}

///////////////////////////////////////////////////////////////////////////////////////////////

} /* namespace cv */
} /* namespace sfm */

#endif

/* End of file. */