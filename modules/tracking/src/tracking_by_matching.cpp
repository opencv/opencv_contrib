// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <utility>
#include <limits>
#include <algorithm>

#include "opencv2/tracking/tracking_by_matching.hpp"
#include "opencv2/core/check.hpp"
#include "kuhn_munkres.hpp"

#define TBM_CHECK(cond) CV_Assert(cond)

#define TBM_CHECK_EQ(actual, expected) CV_CheckEQ(actual, expected, "Assertion error:")
#define TBM_CHECK_NE(actual, expected) CV_CheckNE(actual, expected, "Assertion error:")
#define TBM_CHECK_LT(actual, expected) CV_CheckLT(actual, expected, "Assertion error:")
#define TBM_CHECK_GT(actual, expected) CV_CheckGT(actual, expected, "Assertion error:")
#define TBM_CHECK_LE(actual, expected) CV_CheckLE(actual, expected, "Assertion error:")
#define TBM_CHECK_GE(actual, expected) CV_CheckGE(actual, expected, "Assertion error:")

namespace cv {
namespace detail {
inline namespace tracking {

using namespace tbm;

CosDistance::CosDistance(const cv::Size &descriptor_size)
    : descriptor_size_(descriptor_size) {
    TBM_CHECK(descriptor_size.area() != 0);
}

float CosDistance::compute(const cv::Mat &descr1, const cv::Mat &descr2) {
    TBM_CHECK(!descr1.empty());
    TBM_CHECK(!descr2.empty());
    TBM_CHECK(descr1.size() == descriptor_size_);
    TBM_CHECK(descr2.size() == descriptor_size_);

    double xy = descr1.dot(descr2);
    double xx = descr1.dot(descr1);
    double yy = descr2.dot(descr2);
    double norm = sqrt(xx * yy) + 1e-6;
    return 0.5f * static_cast<float>(1.0 - xy / norm);
}

std::vector<float> CosDistance::compute(const std::vector<cv::Mat> &descrs1,
                                        const std::vector<cv::Mat> &descrs2) {
    TBM_CHECK(descrs1.size() != 0);
    TBM_CHECK(descrs1.size() == descrs2.size());

    std::vector<float> distances(descrs1.size(), 1.f);
    for (size_t i = 0; i < descrs1.size(); i++) {
        distances.at(i) = compute(descrs1.at(i), descrs2.at(i));
    }

    return distances;
}


float MatchTemplateDistance::compute(const cv::Mat &descr1,
                                     const cv::Mat &descr2) {
    TBM_CHECK(!descr1.empty() && !descr2.empty());
    TBM_CHECK_EQ(descr1.size(), descr2.size());
    TBM_CHECK_EQ(descr1.type(), descr2.type());
    cv::Mat res;
    cv::matchTemplate(descr1, descr2, res, type_);
    TBM_CHECK(res.size() == cv::Size(1, 1));
    float dist = res.at<float>(0, 0);
    return scale_ * dist + offset_;
}

std::vector<float> MatchTemplateDistance::compute(const std::vector<cv::Mat> &descrs1,
                                                  const std::vector<cv::Mat> &descrs2) {
    std::vector<float> result;
    for (size_t i = 0; i < descrs1.size(); i++) {
        result.push_back(compute(descrs1[i], descrs2[i]));
    }
    return result;
}

namespace {
cv::Point Center(const cv::Rect& rect) {
    return cv::Point((int)(rect.x + rect.width * .5), (int)(rect.y + rect.height * .5));
}

std::vector<cv::Point> Centers(const TrackedObjects &detections) {
    std::vector<cv::Point> centers(detections.size());
    for (size_t i = 0; i < detections.size(); i++) {
        centers[i] = Center(detections[i].rect);
    }
    return centers;
}

inline bool IsInRange(float val, float min, float max) {
    return min <= val && val <= max;
}

inline bool IsInRange(float val, cv::Vec2f range) {
    return IsInRange(val, range[0], range[1]);
}

std::vector<cv::Scalar> GenRandomColors(int colors_num) {
    std::vector<cv::Scalar> colors(colors_num);
    for (int i = 0; i < colors_num; i++) {
        colors[i] = cv::Scalar(static_cast<uchar>(255. * rand() / RAND_MAX),  // NOLINT
                               static_cast<uchar>(255. * rand() / RAND_MAX),  // NOLINT
                               static_cast<uchar>(255. * rand() / RAND_MAX));  // NOLINT
    }
    return colors;
}

///
/// \brief Draws a polyline on a frame.
/// \param[in] polyline Vector of points (polyline).
/// \param[in] color Color (BGR).
/// \param[in,out] image Frame.
/// \param[in] lwd Line width.
///
void DrawPolyline(const std::vector<cv::Point>& polyline,
                  const cv::Scalar& color, CV_OUT cv::Mat& image,
                  int lwd = 5) {
    TBM_CHECK(!image.empty());
    TBM_CHECK_EQ(image.type(), CV_8UC3);
    TBM_CHECK_GT(lwd, 0);
    TBM_CHECK_LT(lwd, 20);

    for (size_t i = 1; i < polyline.size(); i++) {
        cv::line(image, polyline[i - 1], polyline[i], color, lwd);
    }
}

void ValidateParams(const TrackerParams &p) {
    TBM_CHECK_GE(p.min_track_duration, static_cast<size_t>(500));
    TBM_CHECK_LE(p.min_track_duration, static_cast<size_t>(10000));

    TBM_CHECK_LE(p.forget_delay, static_cast<size_t>(10000));

    TBM_CHECK_GE(p.aff_thr_fast, 0.0f);
    TBM_CHECK_LE(p.aff_thr_fast, 1.0f);

    TBM_CHECK_GE(p.aff_thr_strong, 0.0f);
    TBM_CHECK_LE(p.aff_thr_strong, 1.0f);

    TBM_CHECK_GE(p.shape_affinity_w, 0.0f);
    TBM_CHECK_LE(p.shape_affinity_w, 100.0f);

    TBM_CHECK_GE(p.motion_affinity_w, 0.0f);
    TBM_CHECK_LE(p.motion_affinity_w, 100.0f);

    TBM_CHECK_GE(p.time_affinity_w, 0.0f);
    TBM_CHECK_LE(p.time_affinity_w, 100.0f);

    TBM_CHECK_GE(p.min_det_conf, 0.0f);
    TBM_CHECK_LE(p.min_det_conf, 1.0f);

    TBM_CHECK_GE(p.bbox_aspect_ratios_range[0], 0.0f);
    TBM_CHECK_LE(p.bbox_aspect_ratios_range[1], 10.0f);
    TBM_CHECK_LT(p.bbox_aspect_ratios_range[0], p.bbox_aspect_ratios_range[1]);

    TBM_CHECK_GE(p.bbox_heights_range[0], 10.0f);
    TBM_CHECK_LE(p.bbox_heights_range[1], 1080.0f);
    TBM_CHECK_LT(p.bbox_heights_range[0], p.bbox_heights_range[1]);

    TBM_CHECK_GE(p.predict, 0);
    TBM_CHECK_LE(p.predict, 10000);

    TBM_CHECK_GE(p.strong_affinity_thr, 0.0f);
    TBM_CHECK_LE(p.strong_affinity_thr, 1.0f);

    TBM_CHECK_GE(p.reid_thr, 0.0f);
    TBM_CHECK_LE(p.reid_thr, 1.0f);


    if (p.max_num_objects_in_track > 0) {
        int min_required_track_length = static_cast<int>(p.forget_delay);
        TBM_CHECK_GE(p.max_num_objects_in_track, min_required_track_length);
        TBM_CHECK_LE(p.max_num_objects_in_track, 10000);
    }
}

}  // anonymous namespace

///
/// \brief Tracker-by-Matching algorithm implementation.
///
/// This class is implementation of tracking-by-matching system. It uses two
/// different appearance measures to compute affinity between bounding boxes:
/// some fast descriptor and some strong descriptor. Each time the assignment
/// problem is solved. The assignment problem in our case is how to establish
/// correspondence between existing tracklets and recently detected objects.
/// First step is to compute an affinity matrix between tracklets and
/// detections. The affinity equals to
///       appearance_affinity * motion_affinity * shape_affinity.
/// Where appearance is 1 - distance(tracklet_fast_dscr, detection_fast_dscr).
/// Second step is to solve the assignment problem using Kuhn-Munkres
/// algorithm. If correspondence between some tracklet and detection is
/// established with low confidence (affinity) then the strong descriptor is
/// used to determine if there is correspondence between tracklet and detection.
///
class TrackerByMatching: public ITrackerByMatching {
public:
    using Descriptor = std::shared_ptr<IImageDescriptor>;
    using Distance = std::shared_ptr<IDescriptorDistance>;

    ///
    /// \brief Constructor that creates an instance of the tracker with
    /// parameters.
    /// \param[in] params - the tracker parameters.
    ///
    explicit TrackerByMatching(const TrackerParams &params = TrackerParams());
    virtual ~TrackerByMatching() {}

    ///
    /// \brief Process given frame.
    /// \param[in] frame Colored image (CV_8UC3).
    /// \param[in] detections Detected objects on the frame.
    /// \param[in] timestamp Timestamp must be positive and measured in
    /// milliseconds
    ///
    void process(const cv::Mat &frame, const TrackedObjects &detections,
                 uint64_t timestamp) override;

    ///
    /// \brief Pipeline parameters getter.
    /// \return Parameters of pipeline.
    ///
    const TrackerParams &params() const override;

    ///
    /// \brief Pipeline parameters setter.
    /// \param[in] params Parameters of pipeline.
    ///
    void setParams(const TrackerParams &params) override;

    ///
    /// \brief Fast descriptor getter.
    /// \return Fast descriptor used in pipeline.
    ///
    const Descriptor &descriptorFast() const override;

    ///
    /// \brief Fast descriptor setter.
    /// \param[in] val Fast descriptor used in pipeline.
    ///
    void setDescriptorFast(const Descriptor &val) override;

    ///
    /// \brief Strong descriptor getter.
    /// \return Strong descriptor used in pipeline.
    ///
    const Descriptor &descriptorStrong() const override;

    ///
    /// \brief Strong descriptor setter.
    /// \param[in] val Strong descriptor used in pipeline.
    ///
    void setDescriptorStrong(const Descriptor &val) override;

    ///
    /// \brief Fast distance getter.
    /// \return Fast distance used in pipeline.
    ///
    const Distance &distanceFast() const override;

    ///
    /// \brief Fast distance setter.
    /// \param[in] val Fast distance used in pipeline.
    ///
    void setDistanceFast(const Distance &val) override;

    ///
    /// \brief Strong distance getter.
    /// \return Strong distance used in pipeline.
    ///
    const Distance &distanceStrong() const override;

    ///
    /// \brief Strong distance setter.
    /// \param[in] val Strong distance used in pipeline.
    ///
    void setDistanceStrong(const Distance &val) override;

    ///
    /// \brief Returns number of counted people.
    /// \return a number of counted people.
    ///
    size_t count() const override;

    ///
    /// \brief Get active tracks to draw
    /// \return Active tracks.
    ///
    std::unordered_map<size_t, std::vector<cv::Point> > getActiveTracks() const override;

    ///
    /// \brief Get tracked detections.
    /// \return Tracked detections.
    ///
    TrackedObjects trackedDetections() const override;

    ///
    /// \brief Draws active tracks on a given frame.
    /// \param[in] frame Colored image (CV_8UC3).
    /// \return Colored image with drawn active tracks.
    ///
    cv::Mat drawActiveTracks(const cv::Mat &frame) override;

    ///
    /// \brief Print confusion matrices of data association classifiers.
    /// It works only in case of loaded detection logs instead of native
    /// detectors.
    ///
    void PrintConfusionMatrices() const;

    ///
    /// \brief isTrackForgotten returns true if track is forgotten.
    /// \param id Track ID.
    /// \return true if track is forgotten.
    ///
    bool isTrackForgotten(size_t id) const override;

    ///
    /// \brief tracks Returns all tracks including forgotten (lost too many frames
    /// ago).
    /// \return Set of tracks {id, track}.
    ///
    const std::unordered_map<size_t, Track> &tracks() const override;

    ///
    /// \brief isTrackValid Checks whether track is valid (duration > threshold).
    /// \param track_id Index of checked track.
    /// \return True if track duration exceeds some predefined value.
    ///
    bool isTrackValid(size_t track_id) const override;

    ///
    /// \brief dropForgottenTracks Removes tracks from memory that were lost too
    /// many frames ago.
    ///
    void dropForgottenTracks() override;

    ///
    /// \brief dropForgottenTrack Check that the track was lost too many frames
    /// ago
    /// and removes it frm memory.
    ///
    void dropForgottenTrack(size_t track_id) override;

private:
    struct Match {
        int frame_idx1;
        int frame_idx2;
        cv::Rect rect1;
        cv::Rect rect2;
        cv::Rect pr_rect1;
        bool pr_label;
        bool gt_label;

        Match() {}

        Match(const TrackedObject &a, const cv::Rect &a_pr_rect,
              const TrackedObject &b, bool pr_label)
            : frame_idx1(a.frame_idx),
            frame_idx2(b.frame_idx),
            rect1(a.rect),
            rect2(b.rect),
            pr_rect1(a_pr_rect),
            pr_label(pr_label),
            gt_label(a.object_id == b.object_id) {
                CV_Assert(frame_idx1 != frame_idx2);
            }
    };


    const ObjectTracks all_tracks(bool valid_only) const;
    // Returns shape affinity.
    static float ShapeAffinity(float w, const cv::Rect &trk, const cv::Rect &det);

    // Returns motion affinity.
    static float MotionAffinity(float w, const cv::Rect &trk,
                                const cv::Rect &det);

    // Returns time affinity.
    static float TimeAffinity(float w, const float &trk, const float &det);

    cv::Rect PredictRect(size_t id, size_t k, size_t s) const;

    cv::Rect PredictRectSmoothed(size_t id, size_t k, size_t s) const;

    cv::Rect PredictRectSimple(size_t id, size_t k, size_t s) const;

    void SolveAssignmentProblem(
        const std::set<size_t> &track_ids, const TrackedObjects &detections,
        const std::vector<cv::Mat> &descriptors,
        CV_OUT std::set<size_t>& unmatched_tracks,
        CV_OUT std::set<size_t>& unmatched_detections,
        CV_OUT std::set<std::tuple<size_t, size_t, float>>& matches);

    void ComputeFastDesciptors(const cv::Mat &frame,
                               const TrackedObjects &detections,
                               CV_OUT std::vector<cv::Mat>& desriptors);

    void ComputeDissimilarityMatrix(const std::set<size_t> &active_track_ids,
                                    const TrackedObjects &detections,
                                    const std::vector<cv::Mat> &fast_descriptors,
                                    CV_OUT cv::Mat& dissimilarity_matrix);

    std::vector<float> ComputeDistances(
        const cv::Mat &frame,
        const TrackedObjects& detections,
        const std::vector<std::pair<size_t, size_t>> &track_and_det_ids,
        CV_OUT std::map<size_t, cv::Mat>& det_id_to_descriptor);

    std::map<size_t, std::pair<bool, cv::Mat>> StrongMatching(
        const cv::Mat &frame,
        const TrackedObjects& detections,
        const std::vector<std::pair<size_t, size_t>> &track_and_det_ids);

    std::vector<std::pair<size_t, size_t>> GetTrackToDetectionIds(
        const std::set<std::tuple<size_t, size_t, float>> &matches);

    float AffinityFast(const cv::Mat &descriptor1, const TrackedObject &obj1,
                       const cv::Mat &descriptor2, const TrackedObject &obj2);

    float Affinity(const TrackedObject &obj1, const TrackedObject &obj2);

    void AddNewTrack(const cv::Mat &frame, const TrackedObject &detection,
                     const cv::Mat &fast_descriptor,
                     const cv::Mat &descriptor_strong = cv::Mat());

    void AddNewTracks(const cv::Mat &frame, const TrackedObjects &detections,
                      const std::vector<cv::Mat> &descriptors_fast);

    void AddNewTracks(const cv::Mat &frame, const TrackedObjects &detections,
                      const std::vector<cv::Mat> &descriptors_fast,
                      const std::set<size_t> &ids);

    void AppendToTrack(const cv::Mat &frame, size_t track_id,
                       const TrackedObject &detection,
                       const cv::Mat &descriptor_fast,
                       const cv::Mat &descriptor_strong);

    bool EraseTrackIfBBoxIsOutOfFrame(size_t track_id);

    bool EraseTrackIfItWasLostTooManyFramesAgo(size_t track_id);

    bool UpdateLostTrackAndEraseIfItsNeeded(size_t track_id);

    void UpdateLostTracks(const std::set<size_t> &track_ids);

    static cv::Mat ConfusionMatrix(const std::vector<Match> &matches);

    const std::set<size_t> &active_track_ids() const;

    // Returns decisions made by heuristic based on fast distance/descriptor and
    // shape, motion and time affinity.
    const std::vector<Match> & base_classifier_matches() const;

    // Returns decisions made by heuristic based on strong distance/descriptor
    // and
    // shape, motion and time affinity.
    const std::vector<Match> &reid_based_classifier_matches() const;

    // Returns decisions made by strong distance/descriptor affinity.
    const std::vector<Match> &reid_classifier_matches() const;

    TrackedObjects FilterDetections(const TrackedObjects &detections) const;
    bool isTrackForgotten(const Track &track) const;

    // Parameters of the pipeline.
    TrackerParams params_;

    // Indexes of active tracks.
    std::set<size_t> active_track_ids_;

    // Descriptor fast (base classifer).
    Descriptor descriptor_fast_;

    // Distance fast (base classifer).
    Distance distance_fast_;

    // Descriptor strong (reid classifier).
    Descriptor descriptor_strong_;

    // Distance strong (reid classifier).
    Distance distance_strong_;

    // All tracks.
    std::unordered_map<size_t, Track> tracks_;

    // Previous frame image.
    cv::Size prev_frame_size_;

    struct pair_hash {
        std::size_t operator()(const std::pair<size_t, size_t> &p) const {
            CV_Assert(p.first < 1e6 && p.second < 1e6);
            return static_cast<size_t>(p.first * 1e6 + p.second);
        }
    };

    // Distance between current active tracks.
    std::unordered_map<std::pair<size_t, size_t>, float, pair_hash> tracks_dists_;

    // Whether collect matches and compute confusion matrices for
    // track-detection
    // association task (base classifier, reid-based classifier,
    // reid-classiifer).
    bool collect_matches_;

    // This vector contains decisions made by
    // fast_apperance-motion-shape affinity model.
    std::vector<Match> base_classifier_matches_;

    // This vector contains decisions made by
    // strong_apperance(cnn-reid)-motion-shape affinity model.
    std::vector<Match> reid_based_classifier_matches_;

    // This vector contains decisions made by
    // strong_apperance(cnn-reid) affinity model only.
    std::vector<Match> reid_classifier_matches_;

    // Number of all current tracks.
    size_t tracks_counter_;

    // Number of dropped valid tracks.
    size_t valid_tracks_counter_;

    cv::Size frame_size_;

    std::vector<cv::Scalar> colors_;

    uint64_t prev_timestamp_;
};

cv::Ptr<ITrackerByMatching> cv::tbm::createTrackerByMatching(const TrackerParams &params)
{
    ITrackerByMatching* ptr = new TrackerByMatching(params);
    return cv::Ptr<ITrackerByMatching>(ptr);
}

TrackerParams::TrackerParams()
    : min_track_duration(1000),
    forget_delay(150),
    aff_thr_fast(0.8f),
    aff_thr_strong(0.75f),
    shape_affinity_w(0.5f),
    motion_affinity_w(0.2f),
    time_affinity_w(0.0f),
    min_det_conf(0.1f),
    bbox_aspect_ratios_range(0.666f, 5.0f),
    bbox_heights_range(40.f, 1000.f),
    predict(25),
    strong_affinity_thr(0.2805f),
    reid_thr(0.61f),
    drop_forgotten_tracks(true),
    max_num_objects_in_track(300) {}

// Returns confusion matrix as:
//   |tp fn|
//   |fp tn|
cv::Mat TrackerByMatching::ConfusionMatrix(const std::vector<Match> &matches) {
    const bool kNegative = false;
    cv::Mat conf_mat(2, 2, CV_32F, cv::Scalar(0));
    for (const auto &m : matches) {
        conf_mat.at<float>(m.gt_label == kNegative, m.pr_label == kNegative)++;
    }

    return conf_mat;
}

TrackerByMatching::TrackerByMatching(const TrackerParams &params)
    : params_(params),
    descriptor_strong_(nullptr),
    distance_strong_(nullptr),
    collect_matches_(true),
    tracks_counter_(0),
    valid_tracks_counter_(0),
    frame_size_(0, 0),
    prev_timestamp_(std::numeric_limits<uint64_t>::max()) {
        ValidateParams(params);
    }

// Pipeline parameters getter.
const TrackerParams &TrackerByMatching::params() const { return params_; }

// Pipeline parameters setter.
void TrackerByMatching::setParams(const TrackerParams &params) {
    ValidateParams(params);
    params_ = params;
}

// Descriptor fast getter.
const TrackerByMatching::Descriptor &TrackerByMatching::descriptorFast() const {
    return descriptor_fast_;
}

// Descriptor fast setter.
void TrackerByMatching::setDescriptorFast(const Descriptor &val) {
    descriptor_fast_ = val;
}

// Descriptor strong getter.
const TrackerByMatching::Descriptor &TrackerByMatching::descriptorStrong() const {
    return descriptor_strong_;
}

// Descriptor strong setter.
void TrackerByMatching::setDescriptorStrong(const Descriptor &val) {
    descriptor_strong_ = val;
}

// Distance fast getter.
const TrackerByMatching::Distance &TrackerByMatching::distanceFast() const { return distance_fast_; }

// Distance fast setter.
void TrackerByMatching::setDistanceFast(const Distance &val) { distance_fast_ = val; }

// Distance strong getter.
const TrackerByMatching::Distance &TrackerByMatching::distanceStrong() const { return distance_strong_; }

// Distance strong setter.
void TrackerByMatching::setDistanceStrong(const Distance &val) { distance_strong_ = val; }

// Returns all tracks including forgotten (lost too many frames ago).
const std::unordered_map<size_t, Track> &
TrackerByMatching::tracks() const {
    return tracks_;
}

// Returns indexes of active tracks only.
const std::set<size_t> &TrackerByMatching::active_track_ids() const {
    return active_track_ids_;
}


// Returns decisions made by heuristic based on fast distance/descriptor and
// shape, motion and time affinity.
const std::vector<TrackerByMatching::Match> &
TrackerByMatching::base_classifier_matches() const {
    return base_classifier_matches_;
}

// Returns decisions made by heuristic based on strong distance/descriptor
// and
// shape, motion and time affinity.
const std::vector<TrackerByMatching::Match> &TrackerByMatching::reid_based_classifier_matches() const {
    return reid_based_classifier_matches_;
}

// Returns decisions made by strong distance/descriptor affinity.
const std::vector<TrackerByMatching::Match> &TrackerByMatching::reid_classifier_matches() const {
    return reid_classifier_matches_;
}

TrackedObjects TrackerByMatching::FilterDetections(
    const TrackedObjects &detections) const {
    TrackedObjects filtered_detections;
    for (const auto &det : detections) {
        float aspect_ratio = static_cast<float>(det.rect.height) / det.rect.width;
        if (det.confidence > params_.min_det_conf &&
            IsInRange(aspect_ratio, params_.bbox_aspect_ratios_range) &&
            IsInRange(static_cast<float>(det.rect.height), params_.bbox_heights_range)) {
            filtered_detections.emplace_back(det);
        }
    }
    return filtered_detections;
}

void TrackerByMatching::SolveAssignmentProblem(
    const std::set<size_t> &track_ids, const TrackedObjects &detections,
    const std::vector<cv::Mat> &descriptors,
    std::set<size_t>& unmatched_tracks, std::set<size_t>& unmatched_detections,
    std::set<std::tuple<size_t, size_t, float>>& matches) {
    unmatched_tracks.clear();
    unmatched_detections.clear();

    TBM_CHECK(!track_ids.empty());
    TBM_CHECK(!detections.empty());
    TBM_CHECK(descriptors.size() == detections.size());
    matches.clear();

    cv::Mat dissimilarity;
    ComputeDissimilarityMatrix(track_ids, detections, descriptors,
                               dissimilarity);

    auto res = KuhnMunkres().Solve(dissimilarity);

    for (size_t i = 0; i < detections.size(); i++) {
        unmatched_detections.insert(i);
    }

    int i = 0;
    for (auto id : track_ids) {
        if (res[i] < detections.size()) {
            matches.emplace(id, res[i], 1 - dissimilarity.at<float>(i, static_cast<int>(res[i])));
        } else {
            unmatched_tracks.insert(id);
        }
        i++;
    }
}

const ObjectTracks TrackerByMatching::all_tracks(bool valid_only) const {
    ObjectTracks all_objects;
    int counter = 0;

    std::set<size_t> sorted_ids;
    for (const auto &pair : tracks()) {
        sorted_ids.emplace(pair.first);
    }

    for (size_t id : sorted_ids) {
        if (!valid_only || isTrackValid(id)) {
            TrackedObjects filtered_objects;
            for (const auto &object : tracks().at(id).objects) {
                filtered_objects.emplace_back(object);
                filtered_objects.back().object_id = counter;
            }
            all_objects.emplace(counter++, filtered_objects);
        }
    }
    return all_objects;
}

cv::Rect TrackerByMatching::PredictRect(size_t id, size_t k,
                                        size_t s) const {
    const auto &track = tracks_.at(id);
    TBM_CHECK(!track.empty());

    if (track.size() == 1) {
        return track[0].rect;
    }

    size_t start_i = track.size() > k ? track.size() - k : 0;
    float width = 0, height = 0;

    for (size_t i = start_i; i < track.size(); i++) {
        width += track[i].rect.width;
        height += track[i].rect.height;
    }

    TBM_CHECK(track.size() - start_i > 0);
    width /= (track.size() - start_i);
    height /= (track.size() - start_i);

    float delim = 0;
    cv::Point2f d(0, 0);

    for (size_t i = start_i + 1; i < track.size(); i++) {
        d += cv::Point2f(Center(track[i].rect) - Center(track[i - 1].rect));
        delim += (track[i].frame_idx - track[i - 1].frame_idx);
    }

    if (delim) {
        d /= delim;
    }

    s += 1;

    cv::Point c = Center(track.back().rect);
    return cv::Rect(static_cast<int>(c.x - width / 2 + d.x * s),
                    static_cast<int>(c.y - height / 2 + d.y * s),
                    static_cast<int>(width),
                    static_cast<int>(height));
}


bool TrackerByMatching::EraseTrackIfBBoxIsOutOfFrame(size_t track_id) {
    if (tracks_.find(track_id) == tracks_.end()) return true;
    auto c = Center(tracks_.at(track_id).predicted_rect);
    if (!prev_frame_size_.empty() &&
        (c.x < 0 || c.y < 0 || c.x > prev_frame_size_.width ||
         c.y > prev_frame_size_.height)) {
        tracks_.at(track_id).lost = params_.forget_delay + 1;
        for (auto id : active_track_ids()) {
            size_t min_id = std::min(id, track_id);
            size_t max_id = std::max(id, track_id);
            tracks_dists_.erase(std::pair<size_t, size_t>(min_id, max_id));
        }
        active_track_ids_.erase(track_id);
        return true;
    }
    return false;
}

bool TrackerByMatching::EraseTrackIfItWasLostTooManyFramesAgo(
    size_t track_id) {
    if (tracks_.find(track_id) == tracks_.end()) return true;
    if (tracks_.at(track_id).lost > params_.forget_delay) {
        for (auto id : active_track_ids()) {
            size_t min_id = std::min(id, track_id);
            size_t max_id = std::max(id, track_id);
            tracks_dists_.erase(std::pair<size_t, size_t>(min_id, max_id));
        }
        active_track_ids_.erase(track_id);

        return true;
    }
    return false;
}

bool TrackerByMatching::UpdateLostTrackAndEraseIfItsNeeded(
    size_t track_id) {
    tracks_.at(track_id).lost++;
    tracks_.at(track_id).predicted_rect =
        PredictRect(track_id, params().predict, tracks_.at(track_id).lost);

    bool erased = EraseTrackIfBBoxIsOutOfFrame(track_id);
    if (!erased) erased = EraseTrackIfItWasLostTooManyFramesAgo(track_id);
    return erased;
}

void TrackerByMatching::UpdateLostTracks(
    const std::set<size_t> &track_ids) {
    for (auto track_id : track_ids) {
        UpdateLostTrackAndEraseIfItsNeeded(track_id);
    }
}

void TrackerByMatching::process(const cv::Mat &frame,
                                const TrackedObjects &input_detections,
                                uint64_t timestamp) {
    if (prev_timestamp_ != std::numeric_limits<uint64_t>::max())
        TBM_CHECK_LT(static_cast<size_t>(prev_timestamp_), static_cast<size_t>(timestamp));

    if (frame_size_ == cv::Size(0, 0)) {
        frame_size_ = frame.size();
    } else {
        TBM_CHECK_EQ(frame_size_, frame.size());
    }

    TrackedObjects detections = FilterDetections(input_detections);
    for (auto &obj : detections) {
        obj.timestamp = timestamp;
    }

    std::vector<cv::Mat> descriptors_fast;
    ComputeFastDesciptors(frame, detections, descriptors_fast);

    auto active_tracks = active_track_ids_;

    if (!active_tracks.empty() && !detections.empty()) {
        std::set<size_t> unmatched_tracks, unmatched_detections;
        std::set<std::tuple<size_t, size_t, float>> matches;

        SolveAssignmentProblem(active_tracks, detections, descriptors_fast,
                               unmatched_tracks,
                               unmatched_detections, matches);

        std::map<size_t, std::pair<bool, cv::Mat>> is_matching_to_track;

        if (distance_strong_) {
            std::vector<std::pair<size_t, size_t>> reid_track_and_det_ids =
                GetTrackToDetectionIds(matches);
            is_matching_to_track = StrongMatching(
                frame, detections, reid_track_and_det_ids);
        }

        for (const auto &match : matches) {
            size_t track_id = std::get<0>(match);
            size_t det_id = std::get<1>(match);
            float conf = std::get<2>(match);

            auto last_det = tracks_.at(track_id).objects.back();
            last_det.rect = tracks_.at(track_id).predicted_rect;

            if (collect_matches_ && last_det.object_id >= 0 &&
                detections[det_id].object_id >= 0) {
                base_classifier_matches_.emplace_back(
                    tracks_.at(track_id).objects.back(), last_det.rect,
                    detections[det_id], conf > params_.aff_thr_fast);
            }

            if (conf > params_.aff_thr_fast) {
                AppendToTrack(frame, track_id, detections[det_id],
                              descriptors_fast[det_id], cv::Mat());
                unmatched_detections.erase(det_id);
            } else {
                if (conf > params_.strong_affinity_thr) {
                    if (distance_strong_ && is_matching_to_track[track_id].first) {
                        AppendToTrack(frame, track_id, detections[det_id],
                                      descriptors_fast[det_id],
                                      is_matching_to_track[track_id].second.clone());
                    } else {
                        if (UpdateLostTrackAndEraseIfItsNeeded(track_id)) {
                            AddNewTrack(frame, detections[det_id], descriptors_fast[det_id],
                                        distance_strong_
                                        ? is_matching_to_track[track_id].second.clone()
                                        : cv::Mat());
                        }
                    }

                    unmatched_detections.erase(det_id);
                } else {
                    unmatched_tracks.insert(track_id);
                }
            }
        }

        AddNewTracks(frame, detections, descriptors_fast, unmatched_detections);
        UpdateLostTracks(unmatched_tracks);

        for (size_t id : active_tracks) {
            EraseTrackIfBBoxIsOutOfFrame(id);
        }
    } else {
        AddNewTracks(frame, detections, descriptors_fast);
        UpdateLostTracks(active_tracks);
    }

    prev_frame_size_ = frame.size();
    if (params_.drop_forgotten_tracks) dropForgottenTracks();

    tracks_dists_.clear();
    prev_timestamp_ = timestamp;
}

void TrackerByMatching::dropForgottenTracks() {
    std::unordered_map<size_t, Track> new_tracks;
    std::set<size_t> new_active_tracks;

    size_t max_id = 0;
    if (!active_track_ids_.empty())
        max_id =
            *std::max_element(active_track_ids_.begin(), active_track_ids_.end());

    const size_t kMaxTrackID = 10000;
    bool reassign_id = max_id > kMaxTrackID;

    size_t counter = 0;
    for (const auto &pair : tracks_) {
        if (!isTrackForgotten(pair.first)) {
            new_tracks.emplace(reassign_id ? counter : pair.first, pair.second);
            new_active_tracks.emplace(reassign_id ? counter : pair.first);
            counter++;

        } else {
            if (isTrackValid(pair.first)) {
                valid_tracks_counter_++;
            }
        }
    }
    tracks_.swap(new_tracks);
    active_track_ids_.swap(new_active_tracks);

    tracks_counter_ = reassign_id ? counter : tracks_counter_;
}

void TrackerByMatching::dropForgottenTrack(size_t track_id) {
    TBM_CHECK(isTrackForgotten(track_id));
    TBM_CHECK(active_track_ids_.count(track_id) == 0);
    tracks_.erase(track_id);
}

float TrackerByMatching::ShapeAffinity(float weight, const cv::Rect &trk,
                                       const cv::Rect &det) {
    float w_dist = static_cast<float>(std::fabs(trk.width - det.width) / (trk.width + det.width));
    float h_dist = static_cast<float>(std::fabs(trk.height - det.height) / (trk.height + det.height));
    return exp(-weight * (w_dist + h_dist));
}

float TrackerByMatching::MotionAffinity(float weight, const cv::Rect &trk,
                                        const cv::Rect &det) {
    float x_dist = static_cast<float>(trk.x - det.x) * (trk.x - det.x) /
        (det.width * det.width);
    float y_dist = static_cast<float>(trk.y - det.y) * (trk.y - det.y) /
        (det.height * det.height);
    return exp(-weight * (x_dist + y_dist));
}

float TrackerByMatching::TimeAffinity(float weight, const float &trk_time,
                                      const float &det_time) {
    return exp(-weight * std::fabs(trk_time - det_time));
}

void TrackerByMatching::ComputeFastDesciptors(
    const cv::Mat &frame, const TrackedObjects &detections,
    std::vector<cv::Mat>& desriptors) {
    desriptors = std::vector<cv::Mat>(detections.size(), cv::Mat());
    for (size_t i = 0; i < detections.size(); i++) {
        descriptor_fast_->compute(frame(detections[i].rect).clone(),
                                  desriptors[i]);
    }
}

void TrackerByMatching::ComputeDissimilarityMatrix(
    const std::set<size_t> &active_tracks, const TrackedObjects &detections,
    const std::vector<cv::Mat> &descriptors_fast,
    cv::Mat& dissimilarity_matrix) {
    cv::Mat am(static_cast<int>(active_tracks.size()), static_cast<int>(detections.size()), CV_32F, cv::Scalar(0));
    int i = 0;
    for (auto id : active_tracks) {
        auto ptr = am.ptr<float>(i);
        for (size_t j = 0; j < descriptors_fast.size(); j++) {
            auto last_det = tracks_.at(id).objects.back();
            last_det.rect = tracks_.at(id).predicted_rect;
            ptr[j] = AffinityFast(tracks_.at(id).descriptor_fast, last_det,
                                  descriptors_fast[j], detections[j]);
        }
        i++;
    }
    dissimilarity_matrix = 1.0 - am;
}

std::vector<float> TrackerByMatching::ComputeDistances(
    const cv::Mat &frame,
    const TrackedObjects& detections,
    const std::vector<std::pair<size_t, size_t>> &track_and_det_ids,
    std::map<size_t, cv::Mat>& det_id_to_descriptor) {
    std::map<size_t, size_t> det_to_batch_ids;
    std::map<size_t, size_t> track_to_batch_ids;

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> descriptors;
    for (size_t i = 0; i < track_and_det_ids.size(); i++) {
        size_t track_id = track_and_det_ids[i].first;
        size_t det_id = track_and_det_ids[i].second;

        if (tracks_.at(track_id).descriptor_strong.empty()) {
            images.push_back(tracks_.at(track_id).last_image);
            descriptors.push_back(cv::Mat());
            track_to_batch_ids[track_id] = descriptors.size() - 1;
        }

        images.push_back(frame(detections[det_id].rect));
        descriptors.push_back(cv::Mat());
        det_to_batch_ids[det_id] = descriptors.size() - 1;
    }

    descriptor_strong_->compute(images, descriptors);

    std::vector<cv::Mat> descriptors1;
    std::vector<cv::Mat> descriptors2;
    for (size_t i = 0; i < track_and_det_ids.size(); i++) {
        size_t track_id = track_and_det_ids[i].first;
        size_t det_id = track_and_det_ids[i].second;

        if (tracks_.at(track_id).descriptor_strong.empty()) {
            tracks_.at(track_id).descriptor_strong =
                descriptors[track_to_batch_ids[track_id]].clone();
        }
        det_id_to_descriptor[det_id] = descriptors[det_to_batch_ids[det_id]];

        descriptors1.push_back(descriptors[det_to_batch_ids[det_id]]);
        descriptors2.push_back(tracks_.at(track_id).descriptor_strong);
    }

    std::vector<float> distances =
        distance_strong_->compute(descriptors1, descriptors2);

    return distances;
}

std::vector<std::pair<size_t, size_t>>
TrackerByMatching::GetTrackToDetectionIds(
    const std::set<std::tuple<size_t, size_t, float>> &matches) {
    std::vector<std::pair<size_t, size_t>> track_and_det_ids;

    for (const auto &match : matches) {
        size_t track_id = std::get<0>(match);
        size_t det_id = std::get<1>(match);
        float conf = std::get<2>(match);
        if (conf < params_.aff_thr_fast && conf > params_.strong_affinity_thr) {
            track_and_det_ids.emplace_back(track_id, det_id);
        }
    }
    return track_and_det_ids;
}

std::map<size_t, std::pair<bool, cv::Mat>>
TrackerByMatching::StrongMatching(
    const cv::Mat &frame,
    const TrackedObjects& detections,
    const std::vector<std::pair<size_t, size_t>> &track_and_det_ids) {
    std::map<size_t, std::pair<bool, cv::Mat>> is_matching;

    if (track_and_det_ids.size() == 0) {
        return is_matching;
    }

    std::map<size_t, cv::Mat> det_ids_to_descriptors;
    std::vector<float> distances =
        ComputeDistances(frame, detections,
                         track_and_det_ids, det_ids_to_descriptors);

    for (size_t i = 0; i < track_and_det_ids.size(); i++) {
        auto reid_affinity = 1.0 - distances[i];

        size_t track_id = track_and_det_ids[i].first;
        size_t det_id = track_and_det_ids[i].second;

        const auto& track = tracks_.at(track_id);
        const auto& detection = detections[det_id];

        auto last_det = track.objects.back();
        last_det.rect = track.predicted_rect;

        float affinity = static_cast<float>(reid_affinity * Affinity(last_det, detection));

        if (collect_matches_ && last_det.object_id >= 0 &&
            detection.object_id >= 0) {
            reid_classifier_matches_.emplace_back(track.objects.back(), last_det.rect,
                                                  detection,
                                                  reid_affinity > params_.reid_thr);

            reid_based_classifier_matches_.emplace_back(
                track.objects.back(), last_det.rect, detection,
                affinity > params_.aff_thr_strong);
        }

        bool is_detection_matching =
            reid_affinity > params_.reid_thr && affinity > params_.aff_thr_strong;

        is_matching[track_id] = std::pair<bool, cv::Mat>(
            is_detection_matching, det_ids_to_descriptors[det_id]);
    }
    return is_matching;
}

void TrackerByMatching::AddNewTracks(
    const cv::Mat &frame, const TrackedObjects &detections,
    const std::vector<cv::Mat> &descriptors_fast) {
    TBM_CHECK(detections.size() == descriptors_fast.size());
    for (size_t i = 0; i < detections.size(); i++) {
        AddNewTrack(frame, detections[i], descriptors_fast[i]);
    }
}

void TrackerByMatching::AddNewTracks(
    const cv::Mat &frame, const TrackedObjects &detections,
    const std::vector<cv::Mat> &descriptors_fast, const std::set<size_t> &ids) {
    TBM_CHECK(detections.size() == descriptors_fast.size());
    for (size_t i : ids) {
        TBM_CHECK(i < detections.size());
        AddNewTrack(frame, detections[i], descriptors_fast[i]);
    }
}

void TrackerByMatching::AddNewTrack(const cv::Mat &frame,
                                    const TrackedObject &detection,
                                    const cv::Mat &descriptor_fast,
                                    const cv::Mat &descriptor_strong) {
    auto detection_with_id = detection;
    detection_with_id.object_id = static_cast<int>(tracks_counter_);
    tracks_.emplace(std::pair<size_t, Track>(
            tracks_counter_,
            Track({detection_with_id}, frame(detection.rect).clone(),
                  descriptor_fast.clone(), descriptor_strong.clone())));

    for (size_t id : active_track_ids_) {
        tracks_dists_.emplace(std::pair<size_t, size_t>(id, tracks_counter_),
                              std::numeric_limits<float>::max());
    }

    active_track_ids_.insert(tracks_counter_);
    tracks_counter_++;
}

void TrackerByMatching::AppendToTrack(const cv::Mat &frame,
                                      size_t track_id,
                                      const TrackedObject &detection,
                                      const cv::Mat &descriptor_fast,
                                      const cv::Mat &descriptor_strong) {
    TBM_CHECK(!isTrackForgotten(track_id));

    auto detection_with_id = detection;
    detection_with_id.object_id = static_cast<int>(track_id);

    auto &cur_track = tracks_.at(track_id);
    cur_track.objects.emplace_back(detection_with_id);
    cur_track.predicted_rect = detection.rect;
    cur_track.lost = 0;
    cur_track.last_image = frame(detection.rect).clone();
    cur_track.descriptor_fast = descriptor_fast.clone();
    cur_track.length++;

    if (cur_track.descriptor_strong.empty()) {
        cur_track.descriptor_strong = descriptor_strong.clone();
    } else if (!descriptor_strong.empty()) {
        cur_track.descriptor_strong =
            0.5 * (descriptor_strong + cur_track.descriptor_strong);
    }


    if (params_.max_num_objects_in_track > 0) {
        while (cur_track.size() >
               static_cast<size_t>(params_.max_num_objects_in_track)) {
            cur_track.objects.erase(cur_track.objects.begin());
        }
    }
}

float TrackerByMatching::AffinityFast(const cv::Mat &descriptor1,
                                      const TrackedObject &obj1,
                                      const cv::Mat &descriptor2,
                                      const TrackedObject &obj2) {
    const float eps = static_cast<float>(1e-6);
    float shp_aff = ShapeAffinity(params_.shape_affinity_w, obj1.rect, obj2.rect);
    if (shp_aff < eps) return 0.0;

    float mot_aff =
        MotionAffinity(params_.motion_affinity_w, obj1.rect, obj2.rect);
    if (mot_aff < eps) return 0.0;
    float time_aff =
        TimeAffinity(params_.time_affinity_w, static_cast<float>(obj1.frame_idx), static_cast<float>(obj2.frame_idx));

    if (time_aff < eps) return 0.0;

    float app_aff = static_cast<float>(1.0 - distance_fast_->compute(descriptor1, descriptor2));

    return shp_aff * mot_aff * app_aff * time_aff;
}

float TrackerByMatching::Affinity(const TrackedObject &obj1,
                                  const TrackedObject &obj2) {
    float shp_aff = ShapeAffinity(params_.shape_affinity_w, obj1.rect, obj2.rect);
    float mot_aff =
        MotionAffinity(params_.motion_affinity_w, obj1.rect, obj2.rect);
    float time_aff =
        TimeAffinity(params_.time_affinity_w, static_cast<float>(obj1.frame_idx), static_cast<float>(obj2.frame_idx));
    return shp_aff * mot_aff * time_aff;
}

bool TrackerByMatching::isTrackValid(size_t id) const {
    const auto& track = tracks_.at(id);
    const auto &objects = track.objects;
    if (objects.empty()) {
        return false;
    }
    int64_t duration_ms = objects.back().timestamp - track.first_object.timestamp;
    if (duration_ms < static_cast<int64_t>(params_.min_track_duration))
        return false;
    return true;
}

bool TrackerByMatching::isTrackForgotten(size_t id) const {
    return isTrackForgotten(tracks_.at(id));
}

bool TrackerByMatching::isTrackForgotten(const Track &track) const {
    return (track.lost > params_.forget_delay);
}

size_t TrackerByMatching::count() const {
    size_t count = valid_tracks_counter_;
    for (const auto &pair : tracks_) {
        count += (isTrackValid(pair.first) ? 1 : 0);
    }
    return count;
}

std::unordered_map<size_t, std::vector<cv::Point>>
TrackerByMatching::getActiveTracks() const {
    std::unordered_map<size_t, std::vector<cv::Point>> active_tracks;
    for (size_t idx : active_track_ids()) {
        auto track = tracks().at(idx);
        if (isTrackValid(idx) && !isTrackForgotten(idx)) {
            active_tracks.emplace(idx, Centers(track.objects));
        }
    }
    return active_tracks;
}

TrackedObjects TrackerByMatching::trackedDetections() const {
    TrackedObjects detections;
    for (size_t idx : active_track_ids()) {
        auto track = tracks().at(idx);
        if (isTrackValid(idx) && !track.lost) {
            detections.emplace_back(track.objects.back());
        }
    }
    return detections;
}

cv::Mat TrackerByMatching::drawActiveTracks(const cv::Mat &frame) {
    cv::Mat out_frame = frame.clone();

    if (colors_.empty()) {
        int num_colors = 100;
        colors_ = GenRandomColors(num_colors);
    }

    auto active_tracks = getActiveTracks();
    for (auto active_track : active_tracks) {
        size_t idx = active_track.first;
        auto centers = active_track.second;
        DrawPolyline(centers, colors_[idx % colors_.size()], out_frame);
        std::stringstream ss;
        ss << idx;
        cv::putText(out_frame, ss.str(), centers.back(), cv::FONT_HERSHEY_SCRIPT_COMPLEX, 2.0,
                    colors_[idx % colors_.size()], 3);
        auto track = tracks().at(idx);
        if (track.lost) {
            cv::line(out_frame, active_track.second.back(),
                     Center(track.predicted_rect), cv::Scalar(0, 0, 0), 4);
        }
    }

    return out_frame;
}

const cv::Size kMinFrameSize = cv::Size(320, 240);
const cv::Size kMaxFrameSize = cv::Size(1920, 1080);

void TrackerByMatching::PrintConfusionMatrices() const {
    std::cout << "Base classifier quality: " << std::endl;
    {
        auto cm = ConfusionMatrix(base_classifier_matches());
        std::cout << cm << std::endl;
        std::cout << "or" << std::endl;
        cm.row(0) = cm.row(0) / std::max(1.0, cv::sum(cm.row(0))[0]);
        cm.row(1) = cm.row(1) / std::max(1.0, cv::sum(cm.row(1))[0]);
        std::cout << cm << std::endl << std::endl;
    }

    std::cout << "Reid-based classifier quality: " << std::endl;
    {
        auto cm = ConfusionMatrix(reid_based_classifier_matches());
        std::cout << cm << std::endl;
        std::cout << "or" << std::endl;
        cm.row(0) = cm.row(0) / std::max(1.0, cv::sum(cm.row(0))[0]);
        cm.row(1) = cm.row(1) / std::max(1.0, cv::sum(cm.row(1))[0]);
        std::cout << cm << std::endl << std::endl;
    }

    std::cout << "Reid only classifier quality: " << std::endl;
    {
        auto cm = ConfusionMatrix(reid_classifier_matches());
        std::cout << cm << std::endl;
        std::cout << "or" << std::endl;
        cm.row(0) = cm.row(0) / std::max(1.0, cv::sum(cm.row(0))[0]);
        cm.row(1) = cm.row(1) / std::max(1.0, cv::sum(cm.row(1))[0]);
        std::cout << cm << std::endl << std::endl;
    }
}

}}}  // namespace
