/*
 * Copyright (c) 2011,2012. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */
#include "precomp.hpp"
#include "face_basic.hpp"

namespace cv { namespace face {

// Belhumeur, P. N., Hespanha, J., and Kriegman, D. "Eigenfaces vs. Fisher-
// faces: Recognition using class specific linear projection.". IEEE
// Transactions on Pattern Analysis and Machine Intelligence 19, 7 (1997),
// 711–720.
class Fisherfaces: public BasicFaceRecognizerImpl
{
public:
    // Initializes an empty Fisherfaces model.
    Fisherfaces(int num_components = 0, double threshold = DBL_MAX)
        : BasicFaceRecognizerImpl(num_components, threshold)
    { }

    // Computes a Fisherfaces model with images in src and corresponding labels
    // in labels.
    void train(InputArrayOfArrays src, InputArray labels);

    // Send all predict results to caller side for custom result handling
    void predict(InputArray src, Ptr<PredictCollector> collector, const int state) const;
};

// Removes duplicate elements in a given vector.
template<typename _Tp>
inline std::vector<_Tp> remove_dups(const std::vector<_Tp>& src) {
    typedef typename std::set<_Tp>::const_iterator constSetIterator;
    typedef typename std::vector<_Tp>::const_iterator constVecIterator;
    std::set<_Tp> set_elems;
    for (constVecIterator it = src.begin(); it != src.end(); ++it)
        set_elems.insert(*it);
    std::vector<_Tp> elems;
    for (constSetIterator it = set_elems.begin(); it != set_elems.end(); ++it)
        elems.push_back(*it);
    return elems;
}

//------------------------------------------------------------------------------
// Fisherfaces
//------------------------------------------------------------------------------
void Fisherfaces::train(InputArrayOfArrays src, InputArray _lbls) {
    if(src.total() == 0) {
        String error_message = format("Empty training data was given. You'll need more than one sample to learn a model.");
        CV_Error(Error::StsBadArg, error_message);
    } else if(_lbls.getMat().type() != CV_32SC1) {
        String error_message = format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _lbls.type());
        CV_Error(Error::StsBadArg, error_message);
    }
    // make sure data has correct size
    if(src.total() > 1) {
        for(int i = 1; i < static_cast<int>(src.total()); i++) {
            if(src.getMat(i-1).total() != src.getMat(i).total()) {
                String error_message = format("In the Fisherfaces method all input samples (training images) must be of equal size! Expected %d pixels, but was %d pixels.", src.getMat(i-1).total(), src.getMat(i).total());
                CV_Error(Error::StsUnsupportedFormat, error_message);
            }
        }
    }
    // get data
    Mat labels = _lbls.getMat();
    Mat data = asRowMatrix(src, CV_64FC1);
    // number of samples
    int N = data.rows;
    // make sure labels are passed in correct shape
    if(labels.total() != (size_t) N) {
        String error_message = format("The number of samples (src) must equal the number of labels (labels)! len(src)=%d, len(labels)=%d.", N, labels.total());
        CV_Error(Error::StsBadArg, error_message);
    } else if(labels.rows != 1 && labels.cols != 1) {
        String error_message = format("Expected the labels in a matrix with one row or column! Given dimensions are rows=%s, cols=%d.", labels.rows, labels.cols);
       CV_Error(Error::StsBadArg, error_message);
    }
    // clear existing model data
    _labels.release();
    _projections.clear();
    // safely copy from cv::Mat to std::vector
    std::vector<int> ll;
    for(unsigned int i = 0; i < labels.total(); i++) {
        ll.push_back(labels.at<int>(i));
    }
    // get the number of unique classes
    int C = (int) remove_dups(ll).size();
    // clip number of components to be a valid number
    if((_num_components <= 0) || (_num_components > (C-1)))
        _num_components = (C-1);
    // perform a PCA and keep (N-C) components
    PCA pca(data, Mat(), PCA::DATA_AS_ROW, (N-C));
    // project the data and perform a LDA on it
    LDA lda(pca.project(data),labels, _num_components);
    // store the total mean vector
    _mean = pca.mean.reshape(1,1);
    // store labels
    _labels = labels.clone();
    // store the eigenvalues of the discriminants
    lda.eigenvalues().convertTo(_eigenvalues, CV_64FC1);
    // Now calculate the projection matrix as pca.eigenvectors * lda.eigenvectors.
    // Note: OpenCV stores the eigenvectors by row, so we need to transpose it!
    gemm(pca.eigenvectors, lda.eigenvectors(), 1.0, Mat(), 0.0, _eigenvectors, GEMM_1_T);
    // store the projections of the original data
    for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++) {
        Mat p = LDA::subspaceProject(_eigenvectors, _mean, data.row(sampleIdx));
        _projections.push_back(p);
    }
}

void Fisherfaces::predict(InputArray _src, Ptr<PredictCollector> collector, const int state) const {
    Mat src = _src.getMat();
    // check data alignment just for clearer exception messages
    if(_projections.empty()) {
        // throw error if no data (or simply return -1?)
        String error_message = "This Fisherfaces model is not computed yet. Did you call Fisherfaces::train?";
        CV_Error(Error::StsBadArg, error_message);
    } else if(src.total() != (size_t) _eigenvectors.rows) {
        String error_message = format("Wrong input image size. Reason: Training and Test images must be of equal size! Expected an image with %d elements, but got %d.", _eigenvectors.rows, src.total());
        CV_Error(Error::StsBadArg, error_message);
    }
    // project into LDA subspace
    Mat q = LDA::subspaceProject(_eigenvectors, _mean, src.reshape(1,1));
    // find 1-nearest neighbor
    collector->init((int)_projections.size(), state);
    for (size_t sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) {
        double dist = norm(_projections[sampleIdx], q, NORM_L2);
        int label = _labels.at<int>((int)sampleIdx);
        if (!collector->collect(label, dist, state))return;
    }
}

Ptr<BasicFaceRecognizer> createFisherFaceRecognizer(int num_components, double threshold)
{
    return makePtr<Fisherfaces>(num_components, threshold);
}

} }
