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
#include <set>
#include <limits>
#include <iostream>

namespace cv
{
namespace face
{

// Turk, M., and Pentland, A. "Eigenfaces for recognition.". Journal of
// Cognitive Neuroscience 3 (1991), 71–86.
class Eigenfaces : public BasicFaceRecognizerImpl
{

public:
    // Initializes an empty Eigenfaces model.
    Eigenfaces(int num_components = 0, double threshold = DBL_MAX)
        : BasicFaceRecognizerImpl(num_components, threshold)
    {}

    // Computes an Eigenfaces model with images in src and corresponding labels
    // in labels.
    void train(InputArrayOfArrays src, InputArray labels);

    // Send all predict results to caller side for custom result handling
    void predict(InputArray src, Ptr<PredictCollector> collector, const int state) const;
};

//------------------------------------------------------------------------------
// Eigenfaces
//------------------------------------------------------------------------------
void Eigenfaces::train(InputArrayOfArrays _src, InputArray _local_labels) {
    if(_src.total() == 0) {
        String error_message = format("Empty training data was given. You'll need more than one sample to learn a model.");
        CV_Error(Error::StsBadArg, error_message);
    } else if(_local_labels.getMat().type() != CV_32SC1) {
        String error_message = format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _local_labels.type());
        CV_Error(Error::StsBadArg, error_message);
    }
    // make sure data has correct size
    if(_src.total() > 1) {
        for(int i = 1; i < static_cast<int>(_src.total()); i++) {
            if(_src.getMat(i-1).total() != _src.getMat(i).total()) {
                String error_message = format("In the Eigenfaces method all input samples (training images) must be of equal size! Expected %d pixels, but was %d pixels.", _src.getMat(i-1).total(), _src.getMat(i).total());
                CV_Error(Error::StsUnsupportedFormat, error_message);
            }
        }
    }
    // get labels
    Mat labels = _local_labels.getMat();
    // observations in row
    Mat data = asRowMatrix(_src, CV_64FC1);

    // number of samples
   int n = data.rows;
    // assert there are as much samples as labels
    if(static_cast<int>(labels.total()) != n) {
        String error_message = format("The number of samples (src) must equal the number of labels (labels)! len(src)=%d, len(labels)=%d.", n, labels.total());
        CV_Error(Error::StsBadArg, error_message);
    }
    // clear existing model data
    _labels.release();
    _projections.clear();
    // clip number of components to be valid
    if((_num_components <= 0) || (_num_components > n))
        _num_components = n;

    // perform the PCA
    PCA pca(data, Mat(), PCA::DATA_AS_ROW, _num_components);
    // copy the PCA results
    _mean = pca.mean.reshape(1,1); // store the mean vector
    _eigenvalues = pca.eigenvalues.clone(); // eigenvalues by row
    transpose(pca.eigenvectors, _eigenvectors); // eigenvectors by column
    // store labels for prediction
    _labels = labels.clone();
    // save projections
    for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++) {
        Mat p = LDA::subspaceProject(_eigenvectors, _mean, data.row(sampleIdx));
        _projections.push_back(p);
    }
}

void Eigenfaces::predict(InputArray _src, Ptr<PredictCollector> collector, const int state) const {
    // get data
    Mat src = _src.getMat();
    // make sure the user is passing correct data
    if(_projections.empty()) {
        // throw error if no data (or simply return -1?)
        String error_message = "This Eigenfaces model is not computed yet. Did you call Eigenfaces::train?";
        CV_Error(Error::StsError, error_message);
    } else if(_eigenvectors.rows != static_cast<int>(src.total())) {
        // check data alignment just for clearer exception messages
        String error_message = format("Wrong input image size. Reason: Training and Test images must be of equal size! Expected an image with %d elements, but got %d.", _eigenvectors.rows, src.total());
        CV_Error(Error::StsBadArg, error_message);
    }
    // project into PCA subspace
    Mat q = LDA::subspaceProject(_eigenvectors, _mean, src.reshape(1, 1));
    collector->init((int)_projections.size(), state);
    for (size_t sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) {
        double dist = norm(_projections[sampleIdx], q, NORM_L2);
        int label = _labels.at<int>((int)sampleIdx);
        if (!collector->collect(label, dist, state))return;
    }
}

Ptr<BasicFaceRecognizer> createEigenFaceRecognizer(int num_components, double threshold)
{
    return makePtr<Eigenfaces>(num_components, threshold);
}

}
}
