// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2020 by Archit Rungta


// This header file contains some template deductions for type-mapping andconversion

#include <vector>

#include "jlcxx/jlcxx.hpp"
#include "jlcxx/functions.hpp"
#include "jlcxx/stl.hpp"
#include "jlcxx/array.hpp"
#include "jlcxx/tuple.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv_modules.hpp>
#include <type_traits>

#ifdef HAVE_OPENCV_FEATURES2D
#include <opencv2/features2d.hpp>
typedef cv::SimpleBlobDetector::Params SimpleBlobDetector_Params;
typedef cv::AKAZE::DescriptorType AKAZE_DescriptorType;
typedef cv::AgastFeatureDetector::DetectorType AgastFeatureDetector_DetectorType;
typedef cv::FastFeatureDetector::DetectorType FastFeatureDetector_DetectorType;
typedef cv::DescriptorMatcher::MatcherType DescriptorMatcher_MatcherType;
typedef cv::KAZE::DiffusivityType KAZE_DiffusivityType;
typedef cv::ORB::ScoreType ORB_ScoreType;
#endif

#ifdef HAVE_OPENCV_OBJDETECT

#include "opencv2/objdetect.hpp"

typedef cv::HOGDescriptor::HistogramNormType HOGDescriptor_HistogramNormType;
typedef cv::HOGDescriptor::DescriptorStorageFormat HOGDescriptor_DescriptorStorageFormat;

#endif

#ifdef HAVE_OPENCV_FLANN
#include <opencv2/flann.hpp>
typedef cvflann::flann_distance_t cvflann_flann_distance_t;
typedef cvflann::flann_algorithm_t cvflann_flann_algorithm_t;

typedef cv::flann::IndexParams flann_IndexParams;
typedef cv::flann::SearchParams flann_SearchParams;
#endif


#ifdef HAVE_OPENCV_DNN
typedef cv::dnn::DictValue LayerId;
typedef cv::dnn::Backend dnn_Backend;
#endif

using namespace cv;
using namespace std;
using namespace jlcxx;

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

typedef std::vector<Mat> vector_Mat;

#include "jlcv2_types.hpp"
