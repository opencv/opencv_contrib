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

#ifndef LIBMV_RECONSTRUCTION_PROJECTIVE_RECONSTRUCTION_H_
#define LIBMV_RECONSTRUCTION_PROJECTIVE_RECONSTRUCTION_H_

#include "libmv/reconstruction/reconstruction.h"

namespace libmv {

// Estimates the projection matrices of the two cameras using the fundamental
// matrix.
// The method:
//    selects common matches of the two images
//    robustly estimates the fundamental matrix
//    if the first image has no camera, it creates the camera and initializes
//       the projection matrix as the world frame
//    else, note that we also set the first projection matrix to the world
//        frame (for the time being)
//    estimates the projection matrix of the second camera from the fundamental
//      matrix
//    creates and adds it to the reconstruction
//    inserts only inliers matches into matches_inliers
// Returns true if the projection matrix has succeed
// Returns false if
//    the number of common matches is less than 7
bool ReconstructFromTwoUncalibratedViews(const Matches &matches,
                                         CameraID image_id1,
                                         CameraID image_id2,
                                         Matches *matches_inliers,
                                         Reconstruction *reconstruction);

// Estimates the projection matrix of the camera using the already reconstructed
// structures.
// The method:
//    selects the tracks that have an already reconstructed structure
//    robustly estimates the camera projection matrix by resection (P)
//    creates and adds the new camera to reconstruction
//    inserts only inliers matches into matches_inliers
// Returns true if the resection has succeed
// Returns false if
//    the number of reconstructed Tracks is less than 6
bool UncalibratedCameraResection(const Matches &matches,
                                 CameraID image_id,
                                 Matches *matches_inliers,
                                 Reconstruction *reconstruction);

// This method upgrade the reconstruction into a metric one.
// The method use the linear approach;
//  computes a metric reconstruction from a projective one by computing
//    the dual absolute quadric using linear constraints.
//  estimates the metric rectification H
//  upgrades the reconstruction using H
bool UpgradeToMetric(const Matches &matches,
                     Reconstruction *reconstruction);
}  // namespace libmv

#endif  // LIBMV_RECONSTRUCTION_PROJECTIVE_RECONSTRUCTION_H_
