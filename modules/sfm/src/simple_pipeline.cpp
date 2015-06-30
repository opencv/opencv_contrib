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

#include <opencv2/sfm/simple_pipeline.hpp>

#include "libmv/simple_pipeline/bundle.h"
#include "libmv/simple_pipeline/initialize_reconstruction.h"

#include "libmv/simple_pipeline/tracks.h"


using namespace cv;
using namespace std;


namespace cv
{

void
libmv_solveReconstruction( const libmv::Tracks &tracks,
                           int keyframe1, int keyframe2,
                           double focal_length,
                           double principal_x, double principal_y,
                           double k1, double k2, double k3,
                           libmv_Reconstruction &libmv_reconstruction,
                           int refine_intrinsics )
{
    /* Invert the camera intrinsics. */
    libmv::vector<libmv::Marker> markers = tracks.AllMarkers();
    libmv::EuclideanReconstruction *reconstruction = &libmv_reconstruction.reconstruction;
    libmv::CameraIntrinsics *intrinsics = &libmv_reconstruction.intrinsics;

    intrinsics->SetFocalLength(focal_length, focal_length);
    intrinsics->SetPrincipalPoint(principal_x, principal_y);
    intrinsics->SetRadialDistortion(k1, k2, k3);

    cout << "\tNumber of markers: " << markers.size() << endl;
    for (int i = 0; i < markers.size(); ++i)
    {
        intrinsics->InvertIntrinsics(markers[i].x,
                                     markers[i].y,
                                     &(markers[i].x),
                                     &(markers[i].y));
    }

    libmv::Tracks normalized_tracks(markers);

    cout << "\tframes to init from: " << keyframe1 << " " << keyframe2 << endl;
    libmv::vector<libmv::Marker> keyframe_markers =
        normalized_tracks.MarkersForTracksInBothImages(keyframe1, keyframe2);
    cout << "\tNumber of markers for init: " << keyframe_markers.size() << endl;

    libmv::EuclideanReconstructTwoFrames(keyframe_markers, reconstruction);
    libmv::EuclideanBundle(normalized_tracks, reconstruction);
    libmv::EuclideanCompleteReconstruction(libmv::ReconstructionOptions(), normalized_tracks, reconstruction);

    if (refine_intrinsics)
    {
      libmv::EuclideanBundleCommonIntrinsics( tracks, refine_intrinsics, libmv::BUNDLE_NO_CONSTRAINTS, reconstruction, intrinsics);
    }

    libmv_reconstruction.tracks = tracks;
    libmv_reconstruction.error = libmv::EuclideanReprojectionError(tracks, *reconstruction, *intrinsics);
}

void
parser_2D_tracks( const std::vector<cv::Mat> &points2d, libmv::Tracks &tracks )
{
  const int nframes = static_cast<int>(points2d.size());

  for (int frame = 1; frame <= nframes; ++frame)
  {
    const int ntracks = points2d[frame-1].cols;
    for (int track = 1; track <= ntracks; ++track)
    {
        const Vec2d track_pt = points2d[frame-1].col(track-1);
        tracks.Insert(frame, track, track_pt[0], track_pt[1]);
    }
  }

}

} // namespace cv
