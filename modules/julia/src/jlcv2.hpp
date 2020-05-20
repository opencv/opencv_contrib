
// This header file contains some template deductions for type-mapping andconversion

#include <vector>

#include "jlcxx/jlcxx.hpp"
#include "jlcxx/functions.hpp"
#include "jlcxx/stl.hpp"
#include "jlcxx/array.hpp"
#include "jlcxx/tuple.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <type_traits>


using namespace cv;
using namespace std;
using namespace jlcxx;

#ifdef HAVE_OPENCV_FEATURES2D
typedef SimpleBlobDetector::Params SimpleBlobDetector_Params;
typedef AKAZE::DescriptorType AKAZE_DescriptorType;
typedef AgastFeatureDetector::DetectorType AgastFeatureDetector_DetectorType;
typedef FastFeatureDetector::DetectorType FastFeatureDetector_DetectorType;
typedef DescriptorMatcher::MatcherType DescriptorMatcher_MatcherType;
typedef KAZE::DiffusivityType KAZE_DiffusivityType;
typedef ORB::ScoreType ORB_ScoreType;
#endif

#ifdef HAVE_OPENCV_OBJDETECT

#include "opencv2/objdetect.hpp"

typedef HOGDescriptor::HistogramNormType HOGDescriptor_HistogramNormType;
typedef HOGDescriptor::DescriptorStorageFormat HOGDescriptor_DescriptorStorageFormat;

#endif

#ifdef HAVE_OPENCV_FLANN
typedef cvflann::flann_distance_t cvflann_flann_distance_t;
typedef cvflann::flann_algorithm_t cvflann_flann_algorithm_t;

typedef flann::IndexParams flann_IndexParams;
typedef flann::SearchParams flann_SearchParams;
#endif

template <typename C>
struct get_template_type;
template <typename C>
struct get_template_type_vec;

template <template <typename> class C, typename T>
struct get_template_type<C<T>> {
  using type = T;
};

template <template <typename, int> class C, typename T, int N>
struct get_template_type_vec<C<T, N>> {
  using type = T;
  int dim = N;
};

template<typename T, bool v>
struct force_enum{};
template<typename T>
struct force_enum<T, false>{
  using Type = T;
};
template<typename T>
struct force_enum<T, true>{
  using Type = int;
};

template<typename T>
struct force_enum_int{
  using Type = typename force_enum<T, std::is_enum<T>::value>::Type;
};

#include "jlcv2_types.hpp"