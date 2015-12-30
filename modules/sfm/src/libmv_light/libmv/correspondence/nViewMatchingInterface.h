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

#ifndef LIBMV_CORRESPONDENCE_N_VIEW_MATCHING_INTERFACE_H_
#define LIBMV_CORRESPONDENCE_N_VIEW_MATCHING_INTERFACE_H_

#include <string>

namespace libmv {
namespace correspondence  {

  using namespace std;

class nViewMatchingInterface {

  public:
  virtual ~nViewMatchingInterface() {};

  /**
   * Compute the data and store it in the class map<string,T>
   *
   * \param[in] filename   The file from which the data will be extracted.
   *
   * \return True if success.
   */
  virtual bool computeData(const string & filename)=0;

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
  virtual bool MatchData(const string & dataA, const string & dataB)=0;

  /**
  * From a series of element it compute the cross putative match list.
  *
  * \param[in] vec_data The data on which we want compute cross matches.
  *
  * \return True if success (and any matches was found).
  */
  virtual bool computeCrossMatch( const std::vector<string> & vec_data)=0;
};

} // using namespace correspondence
} // using namespace libmv

#endif  // LIBMV_CORRESPONDENCE_N_VIEW_MATCHING_INTERFACE_H_
