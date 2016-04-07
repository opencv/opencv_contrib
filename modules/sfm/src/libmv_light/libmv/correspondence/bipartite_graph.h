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

#ifndef LIBMV_CORRESPONDENCE_INT_BIPARTITE_GRAPH_H_
#define LIBMV_CORRESPONDENCE_INT_BIPARTITE_GRAPH_H_

#include <limits>
#include <map>
#include <set>
#include <cassert>

namespace libmv {

// A bipartite graph with labelled edges.
template<typename T, typename EdgeT>
class BipartiteGraph {
 public:
  typedef std::map<std::pair<T, T>, EdgeT> EdgeMap;

  void Insert(const T &left, const T &right, const EdgeT &edge) {
    left_to_right_[std::make_pair(left, right)] = edge;
    right_to_left_[std::make_pair(right, left)] = edge;
  }
  void Remove(const T &left, const T &right) {
    typename EdgeMap::iterator iter =
     left_to_right_.find(std::make_pair(left, right));
    if (iter != left_to_right_.end())
      left_to_right_.erase(iter);
    iter = right_to_left_.find(std::make_pair(right, left));
    if (iter != right_to_left_.end())
      right_to_left_.erase(iter);
  }

  int NumLeftLeft(T left) const {
    int n = 0;
    typename EdgeMap::const_iterator it;
    for (it = left_to_right_.begin(); it != left_to_right_.end(); ++it) {
      if (it->first.first == left)
        n++;
    }
    return n;
  }

  int NumLeftRight(T right) const {
    int n = 0;
    typename EdgeMap::const_iterator it;
    for (it = left_to_right_.begin(); it != left_to_right_.end(); ++it) {
      if (it->first.second == right)
        n++;
    }
    return n;
  }

  // Erases all the elements.
  // Note that this function does not desallocate pointers
  void Clear() {
    left_to_right_.clear();
    right_to_left_.clear();
  }
  class Range {
   friend class BipartiteGraph<T, EdgeT>;
   public:
    T left()  const { return reversed_ ? it_->first.second : it_->first.first; }
    T right() const { return reversed_ ? it_->first.first  : it_->first.second;}
    EdgeT edge() const { return it_->second; }

    void  operator++() { ++it_; }
    EdgeT operator*()            { return it_->second; }
    operator bool() const  { return it_ != end_; }

   private:
    Range(typename EdgeMap::const_iterator it,
          typename EdgeMap::const_iterator end,
          bool reversed)
      : reversed_(reversed), it_(it), end_(end) {}

    bool reversed_;
    typename EdgeMap::const_iterator it_, end_;
  };

  Range All() const {
    return Range(left_to_right_.begin(), left_to_right_.end(), false);
  }

  Range AllReversed() const {
    return Range(right_to_left_.begin(), right_to_left_.end(), true);
  }

  Range ToLeft(T left) const {
    return Range(left_to_right_.lower_bound(Lower(left)),
                 left_to_right_.upper_bound(Upper(left)), false);
  }

  Range ToRight(T right) const {
    return Range(right_to_left_.lower_bound(Lower(right)),
                 right_to_left_.upper_bound(Upper(right)), true);
  }

  // Find a pointer to the edge, or NULL if not found.
  const EdgeT *Edge(T left, T right) const {
    typename EdgeMap::const_iterator it =
      left_to_right_.find(std::make_pair(left, right));
    if (it != left_to_right_.end()) {
      return &(it->second);
    }
    return NULL;
  }

 private:
  std::pair<T, T> Lower(T first) const {
    return std::make_pair(first, std::numeric_limits<T>::min());
  }
  std::pair<T, T> Upper(T first) const {
    return std::make_pair(first, std::numeric_limits<T>::max());
  }
  EdgeMap left_to_right_;
  EdgeMap right_to_left_;
};

}  // namespace libmv

#endif  // LIBMV_CORRESPONDENCE_BIPARTITE_GRAPH_H_
