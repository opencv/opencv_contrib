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

#include <opencv2/imgcodecs.hpp>

#include "libmv/base/vector_utils.h"
#include "libmv/correspondence/feature.h"
#include "libmv/correspondence/feature_matching.h"
#include "libmv/correspondence/nRobustViewMatching.h"
#include "libmv/multiview/robust_fundamental.h"

using namespace libmv;
using namespace correspondence;
using namespace std;

nRobustViewMatching::nRobustViewMatching(){
#ifdef CV_VERSION_EPOCH
  m_pDescriber = NULL;
#endif
}

nRobustViewMatching::nRobustViewMatching(
    cv::Ptr<cv::FeatureDetector> pDetector,
    cv::Ptr<cv::DescriptorExtractor> pDescriber){
  m_pDetector = pDetector;
  m_pDescriber = pDescriber;
}

/**
 * Compute the data and store it in the class map<string,T>
 *
 * \param[in] filename   The file from which the data will be extracted.
 *
 * \return True if success.
 */
bool nRobustViewMatching::computeData(const string & filename)
{
  cv::Mat im_cv = cv::imread(filename, 0);
  if (im_cv.empty()) {
    LOG(FATAL) << "Failed loading image: " << filename;
    return false;
  }
  else
  {
    libmv::vector<libmv::Feature *> features;
    std::vector<cv::KeyPoint> features_cv;
    m_pDetector->detect( im_cv, features_cv );
    features.resize(features_cv.size());
    for(size_t i=0; i<features_cv.size(); ++i)
      features[i] = new libmv::PointFeature(features_cv[i]);

    cv::Mat descriptors;
    m_pDescriber->compute(im_cv, features_cv, descriptors);

    // Copy data.
    m_ViewData.insert( make_pair(filename,FeatureSet()) );
    FeatureSet & KeypointData = m_ViewData[filename];
    KeypointData.features.resize(descriptors.rows);
    for(int i = 0;i < descriptors.rows; ++i)
    {
      KeypointFeature & feat = KeypointData.features[i];
      descriptors.row(i).copyTo(feat.descriptor);
      *(PointFeature*)(&feat) = *(PointFeature*)features[i];
    }

    DeleteElements(&features);

    return true;
  }
}

/**
* Compute the putative match between data computed from element A and B
*  Store the match data internally in the class
*  map< <string, string> , MatchObject >
*
* \param[in] The name of the filename A (use computed data for this element)
* \param[in] The name of the filename B (use computed data for this element)
*
* \return True if success.
*/
bool nRobustViewMatching::MatchData(const string & dataA, const string & dataB)
{
  // Check input data
  if ( find(m_vec_InputNames.begin(), m_vec_InputNames.end(), dataA)
          == m_vec_InputNames.end() ||
         find(m_vec_InputNames.begin(), m_vec_InputNames.end(), dataB)
          == m_vec_InputNames.end())
  {
    LOG(INFO) << "[nViewMatching::MatchData] "
              << "Could not identify one of the input name.";
    return false;
  }
  if (m_ViewData.find(dataA) == m_ViewData.end() ||
      m_ViewData.find(dataB) == m_ViewData.end())
  {
    LOG(INFO) << "[nViewMatching::MatchData] "
              << "Could not identify data for one of the input name.";
    return false;
  }

  // Computed data exist for the given name
  int iDataA = find(m_vec_InputNames.begin(), m_vec_InputNames.end(), dataA)
                - m_vec_InputNames.begin();
  int iDataB = find(m_vec_InputNames.begin(), m_vec_InputNames.end(), dataB)
                - m_vec_InputNames.begin();

  Matches matches;
  //TODO(pmoulon) make FindCandidatesMatches a parameter.
  FindCandidateMatches_Ratio(m_ViewData[dataA],
                       m_ViewData[dataB],
                       &matches);
  Matches consistent_matches;
  if (computeConstrainMatches(matches,iDataA,iDataB,&consistent_matches))
  {
    matches = consistent_matches;
  }
  if (matches.NumTracks() > 0)
  {
    m_sharedData.insert(
      make_pair(
        make_pair(m_vec_InputNames[iDataA],m_vec_InputNames[iDataB]),
        matches)
      );
  }

  return true;
}

/**
* From a series of element it computes the cross putative match list.
*
* \param[in] vec_data The data on which we want compute cross matches.
*
* \return True if success (and any matches was found).
*/
bool nRobustViewMatching::computeCrossMatch( const std::vector<string> & vec_data)
{
  if (m_pDetector == NULL || m_pDescriber == NULL)  {
    LOG(FATAL) << "Invalid Detector or Describer.";
    return false;
  }

  m_vec_InputNames = vec_data;
  bool bRes = true;
  for (int i=0; i < vec_data.size(); ++i) {
    bRes &= computeData(vec_data[i]);
  }

  bool bRes2 = true;
  for (int i=0; i < vec_data.size(); ++i) {
    for (int j=0; j < i; ++j)
    {
      if (m_ViewData.find(vec_data[i]) != m_ViewData.end() &&
        m_ViewData.find(vec_data[j]) != m_ViewData.end())
      {
        bRes2 &= this->MatchData( vec_data[i], vec_data[j]);
      }
    }
  }
  return bRes2;
}

bool nRobustViewMatching::computeRelativeMatch(
    const std::vector<string>& vec_data) {
  if (m_pDetector == NULL || m_pDescriber == NULL)  {
    LOG(FATAL) << "Invalid Detector or Describer.";
    return false;
  }

  m_vec_InputNames = vec_data;
  bool bRes = true;
  for (int i=0; i < vec_data.size(); ++i) {
    bRes &= computeData(vec_data[i]);
  }

  bool bRes2 = true;
  for (int i=1; i < vec_data.size(); ++i) {
    if (m_ViewData.find(vec_data[i-1]) != m_ViewData.end() &&
        m_ViewData.find(vec_data[i])   != m_ViewData.end())
    {
      bRes2 &= this->MatchData(vec_data[i-1], vec_data[i]);
    }
  }
  // Match the first and the last images (in order to detect loop)
  bRes2 &= this->MatchData(vec_data[0], vec_data[vec_data.size() - 1]);
  return bRes2;
}

/**
* Give the posibility to constrain the matches list.
*
* \param[in] matchIn The input match data between indexA and indexB.
* \param[in] dataAindex The reference index for element A.
* \param[in] dataBindex The reference index for element B.
* \param[out] matchesOut The output match that satisfy the internal constraint.
*
* \return True if success.
*/
bool nRobustViewMatching::computeConstrainMatches(const Matches & matchIn,
                             int dataAindex,
                             int dataBindex,
                             Matches * matchesOut)
{
  if (matchesOut == NULL)
  {
    LOG(INFO) << "[nViewMatching::computeConstrainMatches]"
              << " Could not export constrained matches.";
    return false;
  }
  libmv::vector<Mat> x;
  libmv::vector<int> tracks, images;
  images.push_back(0);
  images.push_back(1);
  PointMatchMatrices(matchIn, images, &tracks, &x);

  libmv::vector<int> inliers;
  Mat3 H;
  // TODO(pmoulon) Make the Correspondence filter a parameter.
  //HomographyFromCorrespondences2PointRobust(x[0], x[1], 0.3, &H, &inliers);
  //HomographyFromCorrespondences4PointRobust(x[0], x[1], 0.3, &H, &inliers);
  //AffineFromCorrespondences2PointRobust(x[0], x[1], 1, &H, &inliers);
  FundamentalFromCorrespondences7PointRobust(x[0], x[1], 1.0, &H, &inliers);

  //TODO(pmoulon) insert an optimization phase.
  // Rerun Robust correspondance on the inliers.
  // it will allow to compute a better model and filter ugly fitting.

  //-- Assert that the output of the model is consistent :
  // As much as the minimal points are inliers.
  if (inliers.size() > 7 * 2) { //2* [nbPoints required by the estimator]
    // If tracks table is empty initialize it
    if (m_featureToTrackTable.size() == 0)  {
      // Build new correspondence graph containing only inliers.
      for (int l = 0; l < inliers.size(); ++l)  {
        const int k = inliers[l];
        m_featureToTrackTable[matchIn.Get(0, tracks[k])] = l;
        m_featureToTrackTable[matchIn.Get(1, tracks[k])] = l;
        m_tracks.Insert(dataAindex, l,
            matchIn.Get(dataBindex, tracks[k]));
        m_tracks.Insert(dataBindex, l,
            matchIn.Get(dataAindex, tracks[k]));
      }
    }
    else  {
      // Else update the tracks
      for (int l = 0; l < inliers.size(); ++l)  {
        const int k = inliers[l];
        map<const Feature*, int>::const_iterator iter =
          m_featureToTrackTable.find(matchIn.Get(1, tracks[k]));

        if (iter!=m_featureToTrackTable.end())  {
          // Add a feature to the existing track
          const int trackIndex = iter->second;
          m_featureToTrackTable[matchIn.Get(0, tracks[k])] = trackIndex;
          m_tracks.Insert(dataAindex, trackIndex,
            matchIn.Get(0, tracks[k]));
        }
        else  {
          // It's a new track
          const int trackIndex = m_tracks.NumTracks();
          m_featureToTrackTable[matchIn.Get(0, tracks[k])] = trackIndex;
          m_featureToTrackTable[matchIn.Get(1, tracks[k])] = trackIndex;
          m_tracks.Insert(dataAindex, trackIndex,
              matchIn.Get(0, tracks[k]));
          m_tracks.Insert(dataBindex, trackIndex,
              matchIn.Get(1, tracks[k]));
        }
      }
    }
    // Export common feature between the two view
    if (matchesOut) {
      Matches & consistent_matches = *matchesOut;
      // Build new correspondence graph containing only inliers.
      for (int l = 0; l < inliers.size(); ++l) {
        int k = inliers[l];
        for (int i = 0; i < 2; ++i) {
          consistent_matches.Insert(images[i], tracks[k],
              matchIn.Get(images[i], tracks[k]));
        }
      }
    }
  }
  return true;
}

