#pragma once

#include <deque>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <map>
#include <tuple>
#include <set>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"


namespace cv {
namespace tbm { //Tracking-by-Matching
///
/// \brief The TrackedObject struct defines properties of detected object.
///
struct CV_EXPORTS TrackedObject {
    cv::Rect rect;       ///< Detected object ROI (zero area if N/A).
    double confidence;   ///< Detection confidence level (-1 if N/A).
    int frame_idx;       ///< Frame index where object was detected (-1 if N/A).
    int object_id;       ///< Unique object identifier (-1 if N/A).
    uint64_t timestamp;  ///< Timestamp in milliseconds.

    ///
    /// \brief Default constructor.
    ///
    TrackedObject()
        : confidence(-1),
        frame_idx(-1),
        object_id(-1),
        timestamp(0) {}

    ///
    /// \brief Constructor with parameters.
    /// \param rect Bounding box of detected object.
    /// \param confidence Confidence of detection.
    /// \param frame_idx Index of frame.
    /// \param object_id Object ID.
    ///
    TrackedObject(const cv::Rect &rect, float confidence, int frame_idx,
                  int object_id)
        : rect(rect),
        confidence(confidence),
        frame_idx(frame_idx),
        object_id(object_id),
        timestamp(0) {}
};

using TrackedObjects = std::deque<TrackedObject>;

bool operator==(const TrackedObject& first, const TrackedObject& second);
bool operator!=(const TrackedObject& first, const TrackedObject& second);
/// (object id, detected objects) pairs collection.
using ObjectTracks = std::unordered_map<int, TrackedObjects>;

///
/// \brief The IImageDescriptor class declares base class for image
/// descriptor.
///
class CV_EXPORTS IImageDescriptor {
public:
    ///
    /// \brief Descriptor size getter.
    /// \return Descriptor size.
    ///
    virtual cv::Size size() const = 0;

    ///
    /// \brief Computes image descriptor.
    /// \param[in] mat Color image.
    /// \param[out] descr Computed descriptor.
    ///
    virtual void Compute(const cv::Mat &mat, cv::Mat *descr) = 0;

    ///
    /// \brief Computes image descriptors in batches.
    /// \param[in] mats Images of interest.
    /// \param[out] descrs Matrices to store the computed descriptors.
    ///
    virtual void Compute(const std::vector<cv::Mat> &mats,
                         std::vector<cv::Mat> *descrs) = 0;

    ///
    /// \brief Prints performance counts for CNN-based descriptors
    ///
    virtual void PrintPerformanceCounts() const {}

    virtual ~IImageDescriptor() {}
};


///
/// \brief Uses resized image as descriptor.
///
class CV_EXPORTS ResizedImageDescriptor : public IImageDescriptor {
public:
    ///
    /// \brief Constructor.
    /// \param[in] descr_size Size of the descriptor (resized image).
    /// \param[in] interpolation Interpolation algorithm.
    ///
    explicit ResizedImageDescriptor(const cv::Size &descr_size,
                                    const cv::InterpolationFlags interpolation)
        : descr_size_(descr_size), interpolation_(interpolation) {
            CV_Assert(descr_size.width > 0);
            CV_Assert(descr_size.height > 0);
        }

    ///
    /// \brief Returns descriptor size.
    /// \return Number of elements in the descriptor.
    ///
    cv::Size size() const override { return descr_size_; }

    ///
    /// \brief Computes image descriptor.
    /// \param[in] mat Frame containing the image of interest.
    /// \param[out] descr Matrix to store the computed descriptor.
    ///
    void Compute(const cv::Mat &mat, cv::Mat *descr) override {
        CV_Assert(descr != nullptr);
        CV_Assert(!mat.empty());
        cv::resize(mat, *descr, descr_size_, 0, 0, interpolation_);
    }

    ///
    /// \brief Computes images descriptors.
    /// \param[in] mats Frames containing images of interest.
    /// \param[out] descrs Matrices to store the computed descriptors.
    //
    void Compute(const std::vector<cv::Mat> &mats,
                 std::vector<cv::Mat> *descrs) override  {
        CV_Assert(descrs != nullptr);
        descrs->resize(mats.size());
        for (size_t i = 0; i < mats.size(); i++)  {
            Compute(mats[i], &(descrs[i]));
        }
    }

private:
    cv::Size descr_size_;

    cv::InterpolationFlags interpolation_;
};


///
/// \brief The IDescriptorDistance class declares an interface for distance
/// computation between reidentification descriptors.
///
class CV_EXPORTS IDescriptorDistance {
public:
    ///
    /// \brief Computes distance between two descriptors.
    /// \param[in] descr1 First descriptor.
    /// \param[in] descr2 Second descriptor.
    /// \return Distance between two descriptors.
    ///
    virtual float Compute(const cv::Mat &descr1, const cv::Mat &descr2) = 0;

    ///
    /// \brief Computes distances between two descriptors in batches.
    /// \param[in] descrs1 Batch of first descriptors.
    /// \param[in] descrs2 Batch of second descriptors.
    /// \return Distances between descriptors.
    ///
    virtual std::vector<float> Compute(const std::vector<cv::Mat> &descrs1,
                                       const std::vector<cv::Mat> &descrs2) = 0;

    virtual ~IDescriptorDistance() {}
};

// TODO: move to cpp, only factory here
///
/// \brief The CosDistance class allows computing cosine distance between two
/// reidentification descriptors.
///
class CV_EXPORTS CosDistance : public IDescriptorDistance {
public:
    ///
    /// \brief CosDistance constructor.
    /// \param[in] descriptor_size Descriptor size.
    ///
    explicit CosDistance(const cv::Size &descriptor_size);

    ///
    /// \brief Computes distance between two descriptors.
    /// \param descr1 First descriptor.
    /// \param descr2 Second descriptor.
    /// \return Distance between two descriptors.
    ///
    float Compute(const cv::Mat &descr1, const cv::Mat &descr2) override;

    ///
    /// \brief Computes distances between two descriptors in batches.
    /// \param[in] descrs1 Batch of first descriptors.
    /// \param[in] descrs2 Batch of second descriptors.
    /// \return Distances between descriptors.
    ///
    std::vector<float> Compute(
        const std::vector<cv::Mat> &descrs1,
        const std::vector<cv::Mat> &descrs2) override;

private:
    cv::Size descriptor_size_;
};



// TODO: move to cpp, only factory here
///
/// \brief Computes distance between images
///        using MatchTemplate function from OpenCV library
///        and its cross-correlation computation method in particular.
///
class CV_EXPORTS MatchTemplateDistance : public IDescriptorDistance {
public:
    ///
    /// \brief Constructs the distance object.
    ///
    /// \param[in] type Method of MatchTemplate function computation.
    /// \param[in] scale Scale parameter for the distance.
    ///            Final distance is computed as:
    ///            scale * distance + offset.
    /// \param[in] offset Offset parameter for the distance.
    ///            Final distance is computed as:
    ///            scale * distance + offset.
    ///
    MatchTemplateDistance(int type = cv::TemplateMatchModes::TM_CCORR_NORMED,
                          float scale = -1, float offset = 1)
        : type_(type), scale_(scale), offset_(offset) {}
    ///
    /// \brief Computes distance between image descriptors.
    /// \param[in] descr1 First image descriptor.
    /// \param[in] descr2 Second image descriptor.
    /// \return Distance between image descriptors.
    ///
    float Compute(const cv::Mat &descr1, const cv::Mat &descr2) override;
    ///
    /// \brief Computes distances between two descriptors in batches.
    /// \param[in] descrs1 Batch of first descriptors.
    /// \param[in] descrs2 Batch of second descriptors.
    /// \return Distances between descriptors.
    ///
    std::vector<float> Compute(const std::vector<cv::Mat> &descrs1,
                               const std::vector<cv::Mat> &descrs2) override;
    virtual ~MatchTemplateDistance() {}

private:
    int type_;      ///< Method of MatchTemplate function computation.
    float scale_;   ///< Scale parameter for the distance. Final distance is
                    /// computed as: scale * distance + offset.
    float offset_;  ///< Offset parameter for the distance. Final distance is
                    /// computed as: scale * distance + offset.
};

///
/// \brief The TrackerParams struct stores parameters of PedestrianTracker
///
struct CV_EXPORTS TrackerParams {
    size_t min_track_duration;  ///< Min track duration in milliseconds.

    size_t forget_delay;  ///< Forget about track if the last bounding box in
                          /// track was detected more than specified number of
                          /// frames ago.

    float aff_thr_fast;  ///< Affinity threshold which is used to determine if
                         /// tracklet and detection should be combined (fast
                         /// descriptor is used).

    float aff_thr_strong;  ///< Affinity threshold which is used to determine if
                           /// tracklet and detection should be combined(strong
                           /// descriptor is used).

    float shape_affinity_w;  ///< Shape affinity weight.

    float motion_affinity_w;  ///< Motion affinity weight.

    float time_affinity_w;  ///< Time affinity weight.

    float min_det_conf;  ///< Min confidence of detection.

    cv::Vec2f bbox_aspect_ratios_range;  ///< Bounding box aspect ratios range.

    cv::Vec2f bbox_heights_range;  ///< Bounding box heights range.

    int predict;  ///< How many frames are used to predict bounding box in case
    /// of lost track.

    float strong_affinity_thr;  ///< If 'fast' confidence is greater than this
                                /// threshold then 'strong' Re-ID approach is
                                /// used.

    float reid_thr;  ///< Affinity threshold for re-identification.

    bool drop_forgotten_tracks;  ///< Drop forgotten tracks. If it's enabled it
                                 /// disables an ability to get detection log.

    int max_num_objects_in_track;  ///< The number of objects in track is
                                   /// restricted by this parameter. If it is negative or zero, the max number of
                                   /// objects in track is not restricted.

    ///
    /// Default constructor.
    ///
    TrackerParams();
};

///
/// \brief The Track struct describes tracks.
///
struct CV_EXPORTS Track {
    ///
    /// \brief Track constructor.
    /// \param objs Detected objects sequence.
    /// \param last_image Image of last image in the detected object sequence.
    /// \param descriptor_fast Fast descriptor.
    /// \param descriptor_strong Strong descriptor (reid embedding).
    ///
    Track(const TrackedObjects &objs, const cv::Mat &last_image,
          const cv::Mat &descriptor_fast, const cv::Mat &descriptor_strong)
        : objects(objs),
        predicted_rect(!objs.empty() ? objs.back().rect : cv::Rect()),
        last_image(last_image),
        descriptor_fast(descriptor_fast),
        descriptor_strong(descriptor_strong),
        lost(0),
        length(1) {
            CV_Assert(!objs.empty());
            first_object = objs[0];
        }

    ///
    /// \brief empty returns if track does not contain objects.
    /// \return true if track does not contain objects.
    ///
    bool empty() const { return objects.empty(); }

    ///
    /// \brief size returns number of detected objects in a track.
    /// \return number of detected objects in a track.
    ///
    size_t size() const { return objects.size(); }

    ///
    /// \brief operator [] return const reference to detected object with
    ///        specified index.
    /// \param i Index of object.
    /// \return const reference to detected object with specified index.
    ///
    const TrackedObject &operator[](size_t i) const { return objects[i]; }

    ///
    /// \brief operator [] return non-const reference to detected object with
    ///        specified index.
    /// \param i Index of object.
    /// \return non-const reference to detected object with specified index.
    ///
    TrackedObject &operator[](size_t i) { return objects[i]; }

    ///
    /// \brief back returns const reference to last object in track.
    /// \return const reference to last object in track.
    ///
    const TrackedObject &back() const {
        CV_Assert(!empty());
        return objects.back();
    }

    ///
    /// \brief back returns non-const reference to last object in track.
    /// \return non-const reference to last object in track.
    ///
    TrackedObject &back() {
        CV_Assert(!empty());
        return objects.back();
    }

    TrackedObjects objects;   ///< Detected objects;
    cv::Rect predicted_rect;  ///< Rectangle that represents predicted position
                              /// and size of bounding box if track has been lost.
    cv::Mat last_image;       ///< Image of last detected object in track.
    cv::Mat descriptor_fast;  ///< Fast descriptor.
    cv::Mat descriptor_strong;  ///< Strong descriptor (reid embedding).
    size_t lost;                ///< How many frames ago track has been lost.

    TrackedObject first_object;  ///< First object in track.
    size_t length;  ///< Length of a track including number of objects that were
                    /// removed from track in order to avoid memory usage growth.
};

class CV_EXPORTS IPedestrianTracker {
public:
    using Descriptor = std::shared_ptr<IImageDescriptor>;
    using Distance = std::shared_ptr<IDescriptorDistance>;

    ///
    /// \brief Constructor that creates an instance of the pedestrian tracker with
    /// parameters.
    /// \param[in] params - the pedestrian tracker parameters.
    ///
    virtual ~IPedestrianTracker() {}

    ///
    /// \brief Process given frame.
    /// \param[in] frame Colored image (CV_8UC3).
    /// \param[in] detections Detected objects on the frame.
    /// \param[in] timestamp Timestamp must be positive and measured in
    /// milliseconds
    ///
    virtual void Process(const cv::Mat &frame, const TrackedObjects &detections,
                         uint64_t timestamp) = 0;

    ///
    /// \brief Pipeline parameters getter.
    /// \return Parameters of pipeline.
    ///
    virtual const TrackerParams &params() const = 0;

    ///
    /// \brief Pipeline parameters setter.
    /// \param[in] params Parameters of pipeline.
    ///
    virtual void set_params(const TrackerParams &params) = 0;

    ///
    /// \brief Fast descriptor getter.
    /// \return Fast descriptor used in pipeline.
    ///
    virtual const Descriptor &descriptor_fast() const = 0;

    ///
    /// \brief Fast descriptor setter.
    /// \param[in] val Fast descriptor used in pipeline.
    ///
    virtual void set_descriptor_fast(const Descriptor &val) = 0;

    ///
    /// \brief Strong descriptor getter.
    /// \return Strong descriptor used in pipeline.
    ///
    virtual const Descriptor &descriptor_strong() const = 0;

    ///
    /// \brief Strong descriptor setter.
    /// \param[in] val Strong descriptor used in pipeline.
    ///
    virtual void set_descriptor_strong(const Descriptor &val) = 0;

    ///
    /// \brief Fast distance getter.
    /// \return Fast distance used in pipeline.
    ///
    virtual const Distance &distance_fast() const = 0;

    ///
    /// \brief Fast distance setter.
    /// \param[in] val Fast distance used in pipeline.
    ///
    virtual void set_distance_fast(const Distance &val) = 0;

    ///
    /// \brief Strong distance getter.
    /// \return Strong distance used in pipeline.
    ///
    virtual const Distance &distance_strong() const = 0;

    ///
    /// \brief Strong distance setter.
    /// \param[in] val Strong distance used in pipeline.
    ///
    virtual void set_distance_strong(const Distance &val) = 0;

    ///
    /// \brief Returns number of counted people.
    /// \return a number of counted people.
    ///
    virtual size_t Count() const = 0;

    ///
    /// \brief Get active tracks to draw
    /// \return Active tracks.
    ///
    virtual std::unordered_map<size_t, std::vector<cv::Point> > GetActiveTracks() const = 0;

    ///
    /// \brief Get tracked detections.
    /// \return Tracked detections.
    ///
    virtual TrackedObjects TrackedDetections() const = 0;

    ///
    /// \brief Draws active tracks on a given frame.
    /// \param[in] frame Colored image (CV_8UC3).
    /// \return Colored image with drawn active tracks.
    ///
    virtual cv::Mat DrawActiveTracks(const cv::Mat &frame) = 0;

    ///
    /// \brief IsTrackForgotten returns true if track is forgotten.
    /// \param id Track ID.
    /// \return true if track is forgotten.
    ///
    virtual bool IsTrackForgotten(size_t id) const = 0;

    ///
    /// \brief tracks Returns all tracks including forgotten (lost too many frames
    /// ago).
    /// \return Set of tracks {id, track}.
    ///
    virtual const std::unordered_map<size_t, Track> &tracks() const = 0;

    ///
    /// \brief IsTrackValid Checks whether track is valid (duration > threshold).
    /// \param track_id Index of checked track.
    /// \return True if track duration exceeds some predefined value.
    ///
    virtual bool IsTrackValid(size_t track_id) const = 0;

    ///
    /// \brief DropForgottenTracks Removes tracks from memory that were lost too
    /// many frames ago.
    ///
    virtual void DropForgottenTracks() = 0;

    ///
    /// \brief DropForgottenTracks Check that the track was lost too many frames
    /// ago
    /// and removes it frm memory.
    ///
    virtual void DropForgottenTrack(size_t track_id) = 0;
};

CV_EXPORTS cv::Ptr<IPedestrianTracker> CreatePedestrianTracker(const TrackerParams &params = TrackerParams());

} // namespace tbm
} // namespace cv
