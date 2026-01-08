// Copyright (c) 2007, 2008 libmv authors.
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

#ifndef LIBMV_CORRESPONDENCE_MATCHES_H_
#define LIBMV_CORRESPONDENCE_MATCHES_H_

#include <algorithm>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "libmv/base/vector.h"
// TODO(julien) use the bipartite_graph_new.h now
#include "libmv/correspondence/bipartite_graph.h"
#include "libmv/logging/logging.h"
#include "libmv/correspondence/feature.h"
#include "libmv/numeric/numeric.h"

namespace libmv {

class Matches {
 public:
  typedef int ImageID;
  typedef int TrackID;
  typedef BipartiteGraph<int, const Feature *> Graph;

  ~Matches();

  // Iterate over features, silently skiping any that are not FeatureT or
  // derived from FeatureT.
  template<typename FeatureT>
  class Features {
   public:
    ImageID           image()    const { return r_.left();  }
    TrackID           track()    const { return r_.right(); }
    const FeatureT *feature()  const {
      return static_cast<const FeatureT *>(r_.edge());
    }
    operator bool() const { return r_; }
    void operator++() { ++r_; Skip(); }
    Features(Graph::Range range) : r_(range) { Skip(); }

   private:
    void Skip() {
      while (r_ && !dynamic_cast<const FeatureT *> (r_.edge())) ++r_;
    }
    Graph::Range r_;
  };
  typedef Features<PointFeature> Points;

  template<typename T>
  Features<T> All() const { return Features<T>(graph_.All()); }

  template<typename T>
  Features<T> AllReversed() const { return Features<T>(graph_.AllReversed()); }

  template<typename T>
  Features<T> InImage(ImageID image) const {
    return Features<T>(graph_.ToLeft(image));
  }

  template<typename T>
  Features<T> InTrack(TrackID track) const {
    return Features<T>(graph_.ToRight(track));
  }

  void PointFeatures2KeyPoints(Features<PointFeature> features,std::vector<cv::KeyPoint> &keypoints)const;
  void KeyPoints(ImageID image,std::vector<cv::KeyPoint> &keypoints)const;
  void MatchesTwo(ImageID image1,ImageID image2,std::vector<cv::DMatch> &matches)const;
  void DrawMatches(ImageID image_id1,const cv::Mat &image1,ImageID image_id2,const cv::Mat &image2, cv::Mat &out)const;

  // Does not take ownership of feature.
  void Insert(ImageID image, TrackID track, const Feature *feature) {
    graph_.Insert(image, track, feature);
    images_.insert(image);
    tracks_.insert(track);
  }

  void Remove(ImageID image, TrackID track) {
    graph_.Remove(image, track);
  }

  // Erases all the elements.
  // Note that this function does not desallocate features
  void Clear() {
    graph_.Clear();
    images_.clear();
    tracks_.clear();
  }
  // Insert all elements of matches (images, tracks, feature) as new data
  void Insert(const Matches &matches) {
    size_t max_images = GetMaxImageID();
    size_t max_tracks = GetMaxTrackID();
    std::map<ImageID, ImageID> new_image_ids;
    std::map<TrackID, TrackID> new_track_ids;
    std::set<ImageID>::const_iterator iter_image;
    std::set<TrackID>::const_iterator iter_track;

    ImageID image_id;
    iter_image = matches.images_.begin();
    for (; iter_image != matches.images_.end(); ++iter_image) {
      image_id = ++max_images;
      new_image_ids[*iter_image] = image_id;
      images_.insert(image_id);
    }
    TrackID track_id;
    iter_track = matches.tracks_.begin();
    for (; iter_track != matches.tracks_.end(); ++iter_track) {
      track_id = ++max_tracks;
      new_track_ids[*iter_track] = track_id;
      tracks_.insert(track_id);
    }
    iter_image = matches.images_.begin();
    for (; iter_image != matches.images_.end(); ++iter_image) {
      iter_track = matches.tracks_.begin();
      for (; iter_track != matches.tracks_.end(); ++iter_track) {
        const Feature * feature = matches.Get(*iter_image, *iter_track);
        image_id = new_image_ids[*iter_image];
        track_id = new_track_ids[*iter_track];
        graph_.Insert(image_id, track_id, feature);
      }
    }
  }
  // Merge common elements add new data (image, track, feature).
  void Merge(const Matches &matches) {
    std::map<TrackID, TrackID> new_track_ids;
    std::set<ImageID>::const_iterator iter_image;
    std::set<TrackID>::const_iterator iter_track;
    //Find non common elements and add them into new_matches
    std::set<ImageID>::const_iterator found_image;
    std::set<TrackID>::const_iterator found_track;
    iter_image = matches.images_.begin();
    for (; iter_image != matches.images_.end(); ++iter_image) {
      found_image = images_.find(*iter_image);
      if (found_image == images_.end()) {
        images_.insert(*iter_image);
      }
      iter_track = matches.tracks_.begin();
      for (; iter_track != matches.tracks_.end(); ++iter_track) {
        found_track = tracks_.find(*iter_track);
        if (found_track == tracks_.end()
          && new_track_ids.find(*iter_track) == new_track_ids.end()) {
          new_track_ids[*iter_track] = *iter_track;
          tracks_.insert(*iter_track);
        }
        const Feature * feature = matches.Get(*iter_image, *iter_track);
        graph_.Insert(*iter_image, *iter_track, feature);
      }
    }
  }

  const Feature *Get(ImageID image, TrackID track) const {
    const Feature *const *f = graph_.Edge(image, track);
    return f ? *f : NULL;
  }

  ImageID GetMaxImageID() const {
    ImageID max_images = -1;
    std::set<ImageID>::const_iterator iter_image =
     std::max_element (images_.begin(), images_.end());
    if (iter_image != images_.end()) {
      max_images = *iter_image;
    }
    return max_images;
  }

  TrackID GetMaxTrackID() const {
    TrackID max_tracks = -1;
    std::set<TrackID>::const_iterator iter_track =
     std::max_element (tracks_.begin(), tracks_.end());
    if (iter_track != tracks_.end()) {
      max_tracks = *iter_track;
    }
    return max_tracks;
  }

  int GetNumberOfMatches(ImageID id1,ImageID id2) const;

  const std::set<ImageID> &get_images() const {
    return images_;
  }
  const std::set<TrackID> &get_tracks() const {
    return tracks_;
  }

  int NumFeatureImage(ImageID image_id) const {
    return graph_.NumLeftLeft(image_id);
  }

  int NumFeatureTrack(TrackID track_id) const {
    return graph_.NumLeftRight(track_id);
  }


  size_t NumTracks() const { return tracks_.size(); }
  size_t NumImages() const { return images_.size(); }

 private:
  Graph graph_;
  std::set<ImageID> images_;
  std::set<TrackID> tracks_;
};


/**
 * Intersect sorted lists. Destroys originals; leaves results as the single
 * entry in sorted_items.
 */
template<typename T>
void Intersect(std::vector< std::vector<T> > *sorted_items) {
  std::vector<T> tmp;
  while (sorted_items->size() > 1) {
    int n = sorted_items->size();
    std::vector<T> &s1 = (*sorted_items)[n - 1];
    std::vector<T> &s2 = (*sorted_items)[n - 2];
    tmp.resize(std::min(s1.size(), s2.size()));
    typename std::vector<T>::iterator it = std::set_intersection(
        s1.begin(), s1.end(), s2.begin(), s2.end(), tmp.begin());
    tmp.resize(int(it - tmp.begin()));
    std::swap(tmp, s2);
    tmp.resize(0);
    sorted_items->pop_back();
  }
}

/**
 * Extract matrices from a set of matches, containing the point locations. Only
 * points for tracks which appear in all images are returned in tracks.
 *
 * \param matches The matches from which to extract the points.
 * \param images  Which images to extract the points from.
 * \param xs      The resulting matrices containing the points. The entries will
 *                match the ordering of images.
 */
inline void TracksInAllImages(const Matches &matches,
                              const vector<Matches::ImageID> &images,
                              vector<Matches::TrackID> *tracks) {
  if (!images.size()) {
    return;
  }
  std::vector<std::vector<Matches::TrackID> > all_tracks;
  all_tracks.resize(images.size());
  for (int i = 0; i < images.size(); ++i) {
    for (Matches::Points r = matches.InImage<PointFeature>(images[i]); r; ++r) {
      all_tracks[i].push_back(r.track());
    }
  }
  Intersect(&all_tracks);
  CHECK(all_tracks.size() == 1);
  for (size_t i = 0; i < all_tracks[0].size(); ++i) {
    tracks->push_back(all_tracks[0][i]);
  }
}

/**
 * Extract matrices from a set of matches, containing the point locations. Only
 * points for tracks which appear in all images are returned in xs. Each output
 * matrix is of size 2 x N, where N is the number of tracks that are in all the
 * images.
 *
 * \param matches The matches from which to extract the points.
 * \param images  Which images to extract the points from.
 * \param xs      The resulting matrices containing the points. The entries will
 *                match the ordering of images.
 */
inline void PointMatchMatrices(const Matches &matches,
                               const vector<Matches::ImageID> &images,
                               vector<Matches::TrackID> *tracks,
                               vector<Mat> *xs) {
  TracksInAllImages(matches, images, tracks);

  xs->resize(images.size());
  for (int i = 0; i < images.size(); ++i) {
    (*xs)[i].resize(2, tracks->size());
    for (int j = 0; j < tracks->size(); ++j) {
      const PointFeature *f = static_cast<const PointFeature *>(
          matches.Get(images[i], (*tracks)[j]));
      (*xs)[i](0, j) = f->x();
      (*xs)[i](1, j) = f->y();
    }
  }
}

inline void TwoViewPointMatchMatrices(const Matches &matches,
                                      Matches::ImageID image_id1,
                                      Matches::ImageID image_id2,
                                      vector<Mat> *xs) {
  vector<Matches::TrackID> tracks;
  vector<Matches::ImageID> images;
  images.push_back(image_id1);
  images.push_back(image_id2);
  PointMatchMatrices(matches, images, &tracks, xs);
}

// Delete the features in a correspondences. Uses const_cast to avoid the
// constness problems. This is more intended for tests than for actual use.
void DeleteMatchFeatures(Matches *matches);

}  // namespace libmv

#endif  // LIBMV_CORRESPONDENCE_MATCHES_H_
