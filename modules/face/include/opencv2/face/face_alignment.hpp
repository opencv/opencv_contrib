// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef __OPENCV_FACE_ALIGNMENT_HPP__
#define __OPENCV_FACE_ALIGNMENT_HPP__

#include "opencv2/face/facemark_train.hpp"

namespace cv{
namespace face{
class CV_EXPORTS_W FacemarkKazemi : public FacemarkTrain
{
public:
    struct CV_EXPORTS Params
    {
        /**
        * \brief Constructor
        */
        Params();
        /// cascade_depth This stores the deapth of cascade used for training.
        unsigned long cascade_depth;
        /// tree_depth This stores the max height of the regression tree built.
        unsigned long tree_depth;
        /// num_trees_per_cascade_level This stores number of trees fit per cascade level.
        unsigned long num_trees_per_cascade_level;
        /// learning_rate stores the learning rate in gradient boosting, also referred as shrinkage.
        float learning_rate;
        /// oversampling_amount stores number of initialisations used to create training samples.
        unsigned long oversampling_amount;
        /// num_test_coordinates stores number of test coordinates.
        unsigned long num_test_coordinates;
        /// lambda stores a value to calculate probability of closeness of two coordinates.
        float lambda;
        /// num_test_splits stores number of random test splits generated.
        unsigned long num_test_splits;
        /// configfile stores the name of the file containing the values of training parameters
        String configfile;
        /// modelfile stores the name of the file containing the Kazemi model
        String modelfile;
        /// faceCascadefile stores the name of the file containing the face cascade model
        String faceCascadefile;
        /// scale to which all the images and the landmarks need to be scaled
        Size scale;
    };
    static Ptr<FacemarkKazemi> create(const FacemarkKazemi::Params &parameters = FacemarkKazemi::Params());
    virtual ~FacemarkKazemi();
};

}} // namespace
#endif
