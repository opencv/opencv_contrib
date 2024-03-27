#ifndef __OPENCV_PERF_FEATURE2D_HPP__
#define __OPENCV_PERF_FEATURE2D_HPP__

#include "perf_precomp.hpp"

namespace opencv_test {

/* configuration for tests of detectors/descriptors. shared between ocl and cpu tests. */

// detectors/descriptors configurations to test
#define DETECTORS_ONLY                                                                  \
    AGAST_DEFAULT, AGAST_5_8, AGAST_7_12d, AGAST_7_12s, AGAST_OAST_9_16

#define DETECTORS_EXTRACTORS                                                            \
    AKAZE_DEFAULT, AKAZE_DESCRIPTOR_KAZE,                                               \
    BRISK_DEFAULT,                                                                      \
    KAZE_DEFAULT

#define CV_ENUM_EXPAND(name, ...) CV_ENUM(name, __VA_ARGS__)

enum Feature2DVals { DETECTORS_ONLY, DETECTORS_EXTRACTORS };
CV_ENUM_EXPAND(Feature2DType, DETECTORS_ONLY, DETECTORS_EXTRACTORS)

typedef tuple<Feature2DType, string> Feature2DType_String_t;
typedef perf::TestBaseWithParam<Feature2DType_String_t> feature2d;

#define TEST_IMAGES testing::Values(\
    "cv/detectors_descriptors_evaluation/images_datasets/leuven/img1.png",\
    "stitching/a3.png", \
    "stitching/s2.jpg")

static inline Ptr<Feature2D> getFeature2D(Feature2DType type)
{
    switch(type) {
    case AGAST_DEFAULT:
        return AgastFeatureDetector::create();
    case AGAST_5_8:
        return AgastFeatureDetector::create(70, true, AgastFeatureDetector::AGAST_5_8);
    case AGAST_7_12d:
        return AgastFeatureDetector::create(70, true, AgastFeatureDetector::AGAST_7_12d);
    case AGAST_7_12s:
        return AgastFeatureDetector::create(70, true, AgastFeatureDetector::AGAST_7_12s);
    case AGAST_OAST_9_16:
        return AgastFeatureDetector::create(70, true, AgastFeatureDetector::OAST_9_16);
    case AKAZE_DEFAULT:
        return AKAZE::create();
    case AKAZE_DESCRIPTOR_KAZE:
        return AKAZE::create(AKAZE::DESCRIPTOR_KAZE);
    case BRISK_DEFAULT:
        return BRISK::create();
    case KAZE_DEFAULT:
        return KAZE::create();
    default:
        return Ptr<Feature2D>();
    }
}

} // namespace

#endif // __OPENCV_PERF_FEATURE2D_HPP__
