#ifndef SLAM_DATA_BOW_VOCABULARY_H
#define SLAM_DATA_BOW_VOCABULARY_H

#include "data/bow_vocabulary_fwd.hpp"

#ifdef USE_DBOW2
#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedVocabulary.h>
#else
#if __has_include(<fbow/vocabulary.h>)
#include <fbow/vocabulary.h>
#elif __has_include(<fbow/fbow.h>)
#include <fbow/fbow.h>
#else
#error "FBoW headers not found: expected <fbow/vocabulary.h> or <fbow/fbow.h>"
#endif
#endif // USE_DBOW2

namespace cv::slam {
namespace data {
namespace bow_vocabulary_util {

float score(bow_vocabulary* bow_vocab, const bow_vector& bow_vec1, const bow_vector& bow_vec2);
void compute_bow(bow_vocabulary* bow_vocab, const cv::Mat& descriptors, bow_vector& bow_vec, bow_feature_vector& bow_feat_vec);
bow_vocabulary* load(std::string path);

}; // namespace bow_vocabulary_util
}; // namespace data
}; // namespace cv::slam

#endif // SLAM_DATA_BOW_VOCABULARY_H
