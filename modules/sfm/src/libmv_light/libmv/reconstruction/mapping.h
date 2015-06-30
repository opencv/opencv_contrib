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

#ifndef LIBMV_RECONSTRUCTION_MAPPING_H_
#define LIBMV_RECONSTRUCTION_MAPPING_H_

#include "libmv/reconstruction/reconstruction.h"

namespace libmv {

// Reconstructs unreconstructed point tracks observed in the image image_id
// using theirs observations (matches) when the instrinsic parameters are known.
// To be reconstructed, the tracks need to be viewed in more than
// minimum_num_views images.
// The method:
//    selects the tracks that haven't been already reconstructed
//    reconstructs the tracks into structures
//    remove outliers (points behind one camera or at infinity)
//    creates and add them in reconstruction
// Returns the number of structures reconstructed and the list of triangulated
// points
uint PointStructureTriangulationCalibrated(
   const Matches &matches,
   CameraID image_id,
   size_t minimum_num_views,
   Reconstruction *reconstruction,
   vector<StructureID> *new_structures_ids = NULL);

// Retriangulates point tracks observed in the image image_id using theirs
// observations (matches)  when the instrinsic parameters are known.
// To be reconstructed, the tracks need to be viewed in more than
// minimum_num_views images.
// The method:
//    selects the tracks that have been already reconstructed
//    reconstructs the tracks into structures
//    remove outliers (points behind one camera or at infinity)
//    updates the coordinates in the reconstruction
// Returns the number of structures retriangulated
uint PointStructureRetriangulationCalibrated(
   const Matches &matches,
   CameraID image_id,
   Reconstruction *reconstruction);

// Reconstructs unreconstructed point tracks observed in the image image_id
// using theirs observations (matches) when the instrinsic param. are unknown.
// To be reconstructed, the tracks need to be viewed in more than
// minimum_num_views images.
// The method:
//    selects the tracks that haven't been already reconstructed
//    reconstructs the tracks into structures
//    remove outliers (contains NaN coord) TODO(julien) do other tests (rms?)
//    creates and add them in reconstruction
// Returns the number of structures reconstructed and the list of triangulated
// points
uint PointStructureTriangulationUncalibrated(
   const Matches &matches,
   CameraID image_id,
   size_t minimum_num_views,
   Reconstruction *reconstruction,
   vector<StructureID> *new_structures_ids = NULL);

// Retriangulates point tracks observed in the image image_id using theirs
// observations (matches)  when the instrinsic parameters are unknown.
// To be reconstructed, the tracks need to be viewed in more than
// minimum_num_views images.
// The method:
//    selects the tracks that have been already reconstructed
//    reconstructs the tracks into structures
//    remove outliers (contains NaN coord) TODO(julien) do other tests (rms?)
//    updates the coordinates in the reconstruction
// Returns the number of structures retriangulated
uint PointStructureRetriangulationUncalibrated(
   const Matches &matches,
   CameraID image_id,
   Reconstruction *reconstruction);
}  // namespace libmv

#endif  // LIBMV_RECONSTRUCTION_MAPPING_H_
