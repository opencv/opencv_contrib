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

#ifndef LIBMV_RECONSTRUCTION_TOOLS_H_
#define LIBMV_RECONSTRUCTION_TOOLS_H_

#include "libmv/reconstruction/reconstruction.h"

namespace libmv {

/// TODO(julien) put this in vector_utils.h?
/// TODO(julien) can we use Eigen::Map?
/// Convert a vector<Tvec> of vectors Tvec to a matrix Tmat
template <typename Tvec, typename Tmat>
inline void VectorToMatrix(vector<Tvec> &vs, Tmat *m) {
  Tvec v;
  m->resize(v.size(), vs.size());
  size_t c = 0;
  for (Tvec * vec = vs.begin(); vec != vs.end(); ++vec) {
    m->col(c) = *vec;
    c++;
  }
}

// Selects only the already reconstructed tracks observed in the image image_id
// and returns a vector of StructureID and their feature coordinates
void SelectExistingPointStructures(const Matches &matches,
                                   CameraID image_id,
                                   const Reconstruction &reconstruction,
                                   vector<StructureID> *structures_ids,
                                   Mat2X *x_image = NULL);

// Selects only the NOT already reconstructed tracks observed in the image
// image_id and returns a vector of StructureID and their feature coordinates
void SelectNonReconstructedPointStructures(const Matches &matches,
                                           CameraID image_id,
                                           const Reconstruction &reconstruction,
                                           vector<StructureID> *structures_ids,
                                           Mat2X *x_image = NULL);

// Recover the position of the selected point structures
void MatrixOfPointStructureCoordinates(
    const vector<StructureID> &structures_ids,
    const Reconstruction &reconstruction,
    Mat4X *X_world);
}  // namespace libmv

#endif  // LIBMV_RECONSTRUCTION_TOOLS_H_
