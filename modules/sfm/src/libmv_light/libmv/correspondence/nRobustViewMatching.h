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

#ifndef LIBMV_CORRESPONDENCE_N_ROBUST_VIEW_MATCHING_INTERFACE_H_
#define LIBMV_CORRESPONDENCE_N_ROBUST_VIEW_MATCHING_INTERFACE_H_

struct FeatureSet;
#include <map>

#include "libmv/correspondence/feature.h"
#include "libmv/correspondence/matches.h"
#include "libmv/correspondence/nViewMatchingInterface.h"

namespace libmv {
namespace correspondence  {

using namespace std;

class nRobustViewMatching :public nViewMatchingInterface  {

  public:
  nRobustViewMatching();
  // Constructor (Specify a detector and a describer interface)
  // The class do not handle memory management over this two parameter.
  nRobustViewMatching(cv::Ptr<cv::FeatureDetector> pDetector,
                      cv::Ptr<cv::DescriptorExtractor> pDescriber);
  //TODO(pmoulon) Add a constructor with a Detector and a Descriptor
  // Add also a Template function to make the match robust..
  ~nRobustViewMatching(){};

  /**
   * Compute the data and store it in the class map<string,T>
   *
   * \param[in] filename   The file from which the data will be extracted.
   *
   * \return True if success.
   */
  bool computeData(const string & filename);

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
  bool MatchData(const string & dataA, const string & dataB);

  /**
  * From a series of element it computes the cross putative match list.
  *
  * \param[in] vec_data The data on which we want compute cross matches.
  *
  * \return True if success (and any matches was found).
  */
  bool computeCrossMatch( const std::vector<string> & vec_data);


  /**
  * From a series of element it computes the incremental putative match list.
  * (only locally, in the relative neighborhood)
  *
  * \param[in] vec_data The data on which we want compute matches.
  *
  * \return True if success (and any matches was found).
  */
  bool computeRelativeMatch( const std::vector<string> & vec_data);

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
  bool computeConstrainMatches(const Matches & matchIn,
                               int dataAindex,
                               int dataBindex,
                               Matches * matchesOut);

  /// Return pairwise correspondence ( geometrically filtered )
  const map< pair<string,string>, Matches> & getSharedData() const
    { return m_sharedData;  }
  /// Return extracted feature over the given image.
  const map<string,FeatureSet> & getViewData() const
    { return m_ViewData;  }
  /// Return detected geometrical consistent matches
  const Matches & getMatches()  const
    { return m_tracks;  }

private :
  /// Input data names
  std::vector<string> m_vec_InputNames;
  /// Data that represent each named element.
  map<string,FeatureSet> m_ViewData;
  /// Matches between element named element <A,B>.
  map< pair<string,string>, Matches> m_sharedData;

  /// LookUpTable to make the crossCorrespondence easier between tracks
  ///   and feature.
  map<const Feature*, int> m_featureToTrackTable;

  /// Matches between all the view.
  Matches m_tracks;

  /// Interface to detect Keypoint.
  cv::Ptr<cv::FeatureDetector> m_pDetector;
  /// Interface to describe Keypoint.
  cv::Ptr<cv::DescriptorExtractor> m_pDescriber;
};

} // using namespace correspondence
} // using namespace libmv

#endif  // LIBMV_CORRESPONDENCE_N_ROBUST_VIEW_MATCHING_INTERFACE_H_
