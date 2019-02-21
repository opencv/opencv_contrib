// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_QUALITY_QUALITYBRISQUE_HPP
#define OPENCV_QUALITY_QUALITYBRISQUE_HPP

#include "qualitybase.hpp"

namespace cv
{
namespace quality
{

/** @brief Custom deleter for QualityBRISQUE internal data */
struct _QualityBRISQUEDeleter
{
    void operator()(void*) const;
};

/**
@brief BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) is a type of No Reference
Image Quality Assessment. It measures score based on extracting Natural Scene Satistics (https://en.wikipedia.org/wiki/Scene_statistics)
and calculating feature vectors. The current implementation uses trained model on TID 2008 Database (http://www.ponomarenko.info/tid2008.htm).
@cite Mittal2 for original paper and @cite Mittal2_software for original implementation
*/
class CV_EXPORTS_W QualityBRISQUE : public QualityBase {
public:

    /** @brief Computes BRISQUE quality score for input images
    @param imgs Images for which to compute quality (should be passed as a vector<Mat> in C++ and list of images in Python)
    @returns Score (averaged over individual scores of all images) ranging from 0 to 100
    (0 denotes the best quality and 100 denotes the poorest quality). The format of the score is: {score, 0., 0., 0.}
    */
    CV_WRAP cv::Scalar compute( InputArrayOfArrays imgs ) CV_OVERRIDE;

    /**
    @brief Create an object which calculates quality
    @param model_file_path cv::String which contains a path to the BRISQUE model data.  If empty, attempts to load from ${OPENCV_DIR}/testdata/contrib/quality/brisque_allmodel.dat
    @param range_file_path cv::String which contains a path to the BRISQUE range data.  If empty, attempts to load from ${OPENCV_DIR}/testdata/contrib/quality/brisque_allrange.dat
    */
    CV_WRAP static Ptr<QualityBRISQUE> create( const cv::String& model_file_path = "", const cv::String& range_file_path = "" );

    // CV_WRAP static cv::Scalar compute_single( const Mat& cmpImg, const cv::String& model_file_path, const cv::String& range_file_path );
    /**
    @brief static method for computing quality
    @param imgs image(s) for which to compute quality (passed as vector<Mat> in C++ and as list of images in Python)
    @param model_file_path cv::String which contains a path to the BRISQUE model data.  If empty, attempts to load from ${OPENCV_DIR}/testdata/contrib/quality/brisque_allmodel.dat
    @param range_file_path cv::String which contains a path to the BRISQUE range data.  If empty, attempts to load from ${OPENCV_DIR}/testdata/contrib/quality/brisque_allrange.dat
    @returns cv::Scalar result of format {std::double score, 0., 0., 0.}. Score ranges from 0 to 100 (100 means worst and 0 means best)
    */
    CV_WRAP static cv::Scalar compute( InputArrayOfArrays imgs, const cv::String& model_file_path, const cv::String& range_file_path );

    /* brief destructor */
    ~QualityBRISQUE() CV_OVERRIDE;

protected:

    /** @brief Internal constructor */
    QualityBRISQUE( const cv::String& model_file_path, const cv::String& range_file_path );

    // type-erased svmmodel
    void* _svm_model = nullptr;

    static constexpr const std::size_t _SVM_RANGE_SIZE = 36U;
    using _svm_range_type = float;

    std::array<_svm_range_type, _SVM_RANGE_SIZE>
        _svm_range_min = {}
        , _svm_range_max = {}
    ;

};  // QualityBRISQUE
}   // quality
}   // cv
#endif
