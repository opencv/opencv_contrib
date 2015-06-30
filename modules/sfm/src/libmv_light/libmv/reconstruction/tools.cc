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

#include <stdio.h>
#include <locale.h>

#include "libmv/reconstruction/tools.h"

namespace libmv {

// Selects only the already reconstructed tracks observed in the image image_id
// and returns a vector of StructureID and their feature coordinates
void SelectExistingPointStructures(const Matches &matches,
                                   CameraID image_id,
                                   const Reconstruction &reconstruction,
                                   vector<StructureID> *structures_ids,
                                   Mat2X *x_image) {
  const size_t kNumberStructuresToReserve = 1000;
  structures_ids->resize(0);
  //TODO(julien) clean this
  structures_ids->reserve(kNumberStructuresToReserve);
  vector<Vec2> xs;
  if (x_image)
    xs.reserve(kNumberStructuresToReserve);
  Matches::Features<PointFeature> fp =
    matches.InImage<PointFeature>(image_id);
  while (fp) {
    if (reconstruction.TrackHasStructure(fp.track())) {
      structures_ids->push_back(fp.track());
      if (x_image)
        xs.push_back(fp.feature()->coords.cast<double>());
    }
    fp.operator++();
  }
  if (x_image)
    VectorToMatrix<Vec2, Mat2X>(xs, x_image);
}

// Selects only the NOT already reconstructed tracks observed in the image
// image_id and returns a vector of StructureID and their feature coordinates
void SelectNonReconstructedPointStructures(const Matches &matches,
                                           CameraID image_id,
                                           const Reconstruction &reconstruction,
                                           vector<StructureID> *structures_ids,
                                           Mat2X *x_image) {
  const size_t kNumberStructuresToReserve = 10000;
  structures_ids->resize(0);
  //TODO(julien) clean this
  structures_ids->reserve(kNumberStructuresToReserve);
  vector<Vec2> xs;
  if (x_image)
    xs.reserve(kNumberStructuresToReserve);
  Matches::Features<PointFeature> fp =
    matches.InImage<PointFeature>(image_id);
  while (fp) {
    if (!reconstruction.TrackHasStructure(fp.track())) {
      structures_ids->push_back(fp.track());
      if (x_image)
        xs.push_back(fp.feature()->coords.cast<double>());
    }
    fp.operator++();
  }
  if (x_image)
    VectorToMatrix<Vec2, Mat2X>(xs, x_image);
}

// Recover the position of the selected point structures
void MatrixOfPointStructureCoordinates(
    const vector<StructureID> &structures_ids,
    const Reconstruction &reconstruction,
    Mat4X *X_world) {
  X_world->resize(4, structures_ids.size());
  PointStructure *point_s = NULL;
  for (size_t s = 0; s < structures_ids.size(); ++s) {
    point_s = dynamic_cast<PointStructure*>(
      reconstruction.GetStructure(structures_ids[s]));
    if (point_s) {
      X_world->col(s) << point_s->coords();
    }
  }
}
} // namespace libmv
