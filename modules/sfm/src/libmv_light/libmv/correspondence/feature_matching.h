// Copyright (c) 2009 libmv authors.
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


#ifndef LIBMV_CORRESPONDENCE_FEATURE_MATCHING_H_
#define LIBMV_CORRESPONDENCE_FEATURE_MATCHING_H_

#include <iostream>

#include <opencv2/core.hpp>

#include "libmv/base/vector.h"
#include "libmv/correspondence/feature.h"
#include "libmv/correspondence/matches.h"

using namespace libmv;

/// Define the description of a feature described by :
/// A PointFeature (x,y,scale,orientation),
/// And a descriptor (a vector of floats).
class KeypointFeature : public ::PointFeature {
  public:
  virtual ~KeypointFeature(){};

  void set(const PointFeature &feature,const cv::Mat &descriptor)
  {
    PointFeature::operator=(feature);
    descriptor.copyTo(this->descriptor);
  }

  // Match kdtree traits: with this, the Feature can act as a kdtree point.
  float operator[](int i) const {
    if (descriptor.depth() != CV_32F)
      std::cerr << "KeypointFeature does not contain floats" << std::endl;
    return descriptor.at<float>(i);
  }

  cv::Mat descriptor;
};

/// FeatureSet : Store an array of KeypointFeature ( Keypoint and descriptor).
struct FeatureSet {
  std::vector<KeypointFeature> features;

  /// return a float * containing the concatenation of descriptor data.
  /// Must be deleted with []
  static cv::Mat FeatureSetDescriptorsToContiguousArray
    ( const FeatureSet & featureSet );
};

// Compute candidate matches between 2 sets of features.  Two features a and b
// are a candidate match if a is the nearest neighbor of b and b is the nearest
// neighbor of a.
void FindCandidateMatches(const FeatureSet &left,
                          const FeatureSet &right,
                          Matches *matches);

// Compute candidate matches between 2 sets of features.
// Keep only strong and distinctive matches by using the Davide Lowe's ratio
// method.
// I.E:  A match is considered as strong if the following test is true :
// I.E distance[0] < fRatio * distances[1].
// From David Lowe “Distinctive Image Features from Scale-Invariant Keypoints”.
// You can use David Lowe's magic ratio (0.6 or 0.8).
// 0.8 allow to remove 90% of the false matches while discarding less than 5%
// of the correct matches.
void FindCandidateMatches_Ratio(const FeatureSet &left,
                          const FeatureSet &right,
                          Matches *matches,
                          float fRatio = 0.8f);
// TODO(pmoulon) Add Lowe's ratio symmetric match method.
// Compute correspondences that match between 2 sets of features with a ratio.

void FindCorrespondences(const FeatureSet &left,
                         const FeatureSet &right,
                         std::map<size_t, size_t> *correspondences,
                         float fRatio = 0.8f);

#endif //LIBMV_CORRESPONDENCE_FEATURE_MATCHING_H_
