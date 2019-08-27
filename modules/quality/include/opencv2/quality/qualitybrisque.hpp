// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_QUALITY_QUALITYBRISQUE_HPP
#define OPENCV_QUALITY_QUALITYBRISQUE_HPP

#include "qualitybase.hpp"
#include "opencv2/ml.hpp"

namespace cv
{
namespace quality
{

/**
@brief BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) is a No Reference Image Quality Assessment (NR-IQA) algorithm.

BRISQUE computes a score based on extracting Natural Scene Statistics (https://en.wikipedia.org/wiki/Scene_statistics)
and calculating feature vectors. See Mittal et al. @cite Mittal2 for original paper and original implementation @cite Mittal2_software .

A trained model is provided in the /samples/ directory and is trained on the LIVE-R2 database @cite Sheikh as in the original implementation.
When evaluated against the TID2008 database @cite Ponomarenko , the SROCC is -0.8424 versus the SROCC of -0.8354 in the original implementation.
C++ code for the BRISQUE LIVE-R2 trainer and TID2008 evaluator are also provided in the /samples/ directory.
*/
class CV_EXPORTS_W QualityBRISQUE : public QualityBase {
public:

    /** @brief Computes BRISQUE quality score for input image
    @param img Image for which to compute quality
    @returns cv::Scalar with the score in the first element.  The score ranges from 0 (best quality) to 100 (worst quality)
    */
    CV_WRAP cv::Scalar compute( InputArray img ) CV_OVERRIDE;

    /**
    @brief Create an object which calculates quality
    @param model_file_path cv::String which contains a path to the BRISQUE model data, eg. /path/to/brisque_model_live.yml
    @param range_file_path cv::String which contains a path to the BRISQUE range data, eg. /path/to/brisque_range_live.yml
    */
    CV_WRAP static Ptr<QualityBRISQUE> create( const cv::String& model_file_path, const cv::String& range_file_path );

    /**
    @brief Create an object which calculates quality
    @param model cv::Ptr<cv::ml::SVM> which contains a loaded BRISQUE model
    @param range cv::Mat which contains BRISQUE range data
    */
    CV_WRAP static Ptr<QualityBRISQUE> create( const cv::Ptr<cv::ml::SVM>& model, const cv::Mat& range );

    /**
    @brief static method for computing quality
    @param img image for which to compute quality
    @param model_file_path cv::String which contains a path to the BRISQUE model data, eg. /path/to/brisque_model_live.yml
    @param range_file_path cv::String which contains a path to the BRISQUE range data, eg. /path/to/brisque_range_live.yml
    @returns cv::Scalar with the score in the first element.  The score ranges from 0 (best quality) to 100 (worst quality)
    */
    CV_WRAP static cv::Scalar compute( InputArray img, const cv::String& model_file_path, const cv::String& range_file_path );

    /**
    @brief static method for computing image features used by the BRISQUE algorithm
    @param img image (BGR(A) or grayscale) for which to compute features
    @param features output row vector of features to cv::Mat or cv::UMat
    */
    CV_WRAP static void computeFeatures(InputArray img, OutputArray features);

protected:

    cv::Ptr<cv::ml::SVM> _model = nullptr;
    cv::Mat _range;

    /** @brief Internal constructor */
    QualityBRISQUE( const cv::String& model_file_path, const cv::String& range_file_path );

    /** @brief Internal constructor */
    QualityBRISQUE(const cv::Ptr<cv::ml::SVM>& model, const cv::Mat& range )
        : _model{ model }
        , _range{ range }
    {}

};  // QualityBRISQUE
}   // quality
}   // cv
#endif
