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

#include <opencv2/features2d.hpp>

#include "libmv/correspondence/feature_matching.h"

// Compute candidate matches between 2 sets of features.  Two features A and B
// are a candidate match if A is the nearest neighbor of B and B is the nearest
// neighbor of A.
void FindCandidateMatches(const FeatureSet &left,
                          const FeatureSet &right,
                          Matches *matches) {
  if (left.features.empty() ||
      right.features.empty() )  {
    return;
  }

  cv::FlannBasedMatcher matcherA;
  cv::FlannBasedMatcher matcherB;

  // Paste the necessary data in contiguous arrays.
  cv::Mat arrayA = FeatureSet::FeatureSetDescriptorsToContiguousArray(left);
  cv::Mat arrayB = FeatureSet::FeatureSetDescriptorsToContiguousArray(right);

  matcherA.add(std::vector<cv::Mat>(1, arrayB));
  matcherB.add(std::vector<cv::Mat>(1, arrayA));
  std::vector<cv::DMatch> matchesA, matchesB;
  matcherA.match(arrayA, matchesA);
  matcherB.match(arrayB, matchesB);

  // From putative matches get symmetric matches.
  int max_track_number = 0;
  for (size_t i = 0; i < matchesA.size(); ++i)
  {
    // Add the match only if we have a symmetric result.
    if (i == matchesB[matchesA[i].trainIdx].trainIdx)
    {
      matches->Insert(0, max_track_number, &left.features[i]);
      matches->Insert(1, max_track_number, &right.features[matchesA[i].trainIdx]);
      ++max_track_number;
    }
  }
}

cv::Mat FeatureSet::FeatureSetDescriptorsToContiguousArray
  ( const FeatureSet & featureSet ) {

  if (featureSet.features.empty())  {
    return cv::Mat();
  }
  int descriptorSize = featureSet.features[0].descriptor.cols;
  // Allocate and paste the necessary data.
  cv::Mat array(featureSet.features.size(), descriptorSize, CV_32F);

  //-- Paste data in the contiguous array :
  for (int i = 0; i < (int)featureSet.features.size(); ++i) {
    featureSet.features[i].descriptor.copyTo(array.row(i));
  }
  return array;
}

// Compute candidate matches between 2 sets of features with a ratio.
void FindCandidateMatches_Ratio(const FeatureSet &left,
                          const FeatureSet &right,
                          Matches *matches,
                          float fRatio) {
  if (left.features.empty() || right.features.empty())
    return;

  cv::FlannBasedMatcher matcherA;

  // Paste the necessary data in contiguous arrays.
  cv::Mat arrayA = FeatureSet::FeatureSetDescriptorsToContiguousArray(left);
  cv::Mat arrayB = FeatureSet::FeatureSetDescriptorsToContiguousArray(right);

  matcherA.add(std::vector<cv::Mat>(1, arrayB));
  std::vector < std::vector<cv::DMatch> > matchesA;
  matcherA.knnMatch(arrayA, matchesA, 2);

  // From putative matches get matches that fit the "Ratio" heuristic.
  int max_track_number = 0;
  for (size_t i = 0; i < matchesA.size(); ++i)
  {
    float distance0 = matchesA[i][0].distance;
    float distance1 = matchesA[i][1].distance;
    // Add the match only if we have a symmetric result.
    if (distance0 < fRatio * distance1)
    {
      {
        matches->Insert(0, max_track_number, &left.features[i]);
        matches->Insert(1, max_track_number, &right.features[matchesA[i][0].trainIdx]);
        ++max_track_number;
      }
    }
  }
}

// Compute correspondences that match between 2 sets of features with a ratio.
void FindCorrespondences(const FeatureSet &left,
                         const FeatureSet &right,
                         std::map<size_t, size_t> *correspondences,
                         float fRatio) {
  if (left.features.empty() || right.features.empty())
    return;

  cv::FlannBasedMatcher matcherA;

  // Paste the necessary data in contiguous arrays.
  cv::Mat arrayA = FeatureSet::FeatureSetDescriptorsToContiguousArray(left);
  cv::Mat arrayB = FeatureSet::FeatureSetDescriptorsToContiguousArray(right);

  matcherA.add(std::vector<cv::Mat>(1, arrayB));
  std::vector < std::vector<cv::DMatch> > matchesA;
  matcherA.knnMatch(arrayA, matchesA, 2);

  // From putative matches get matches that fit the "Ratio" heuristic.
  for (size_t i = 0; i < matchesA.size(); ++i)
  {
    float distance0 = matchesA[i][0].distance;
    float distance1 = matchesA[i][1].distance;
    // Add the match only if we have a symmetric result.
    if (distance0 < fRatio * distance1)
      (*correspondences)[i] = matchesA[i][0].trainIdx;
  }
}
