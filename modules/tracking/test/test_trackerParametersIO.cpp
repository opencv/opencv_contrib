// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/tracking.hpp"

using namespace cv;

TEST(MEDIAN_FLOW_Parameters, IO)
{
    TrackerMedianFlow::Params parameters;

    parameters.maxLevel = 10;
    parameters.maxMedianLengthOfDisplacementDifference = 11;
    parameters.pointsInGrid = 12;
    parameters.winSize = Size(6, 5);
    parameters.winSizeNCC = Size(41, 40);
    parameters.termCriteria.maxCount = 100;
    parameters.termCriteria.epsilon = 0.1;

    FileStorage fsWriter("parameters.xml", FileStorage::WRITE + FileStorage::MEMORY);
    parameters.write(fsWriter);

    String serializedParameters = fsWriter.releaseAndGetString();

    FileStorage fsReader(serializedParameters, FileStorage::READ + FileStorage::MEMORY);

    TrackerMedianFlow::Params readParameters;
    readParameters.read(fsReader.root());

    ASSERT_EQ(parameters.maxLevel, readParameters.maxLevel);
    ASSERT_EQ(parameters.maxMedianLengthOfDisplacementDifference,
              readParameters.maxMedianLengthOfDisplacementDifference);
    ASSERT_EQ(parameters.pointsInGrid, readParameters.pointsInGrid);
    ASSERT_EQ(parameters.winSize, readParameters.winSize);
    ASSERT_EQ(parameters.winSizeNCC, readParameters.winSizeNCC);
    ASSERT_EQ(parameters.termCriteria.epsilon, readParameters.termCriteria.epsilon);
    ASSERT_EQ(parameters.termCriteria.maxCount, readParameters.termCriteria.maxCount);
}


TEST(MEDIAN_FLOW_Parameters, Default_Value_If_Absent)
{
    TrackerMedianFlow::Params defaultParameters;

    FileStorage fsReader(String("%YAML 1.0"), FileStorage::READ + FileStorage::MEMORY);

    TrackerMedianFlow::Params readParameters;
    readParameters.read(fsReader.root());

    ASSERT_EQ(defaultParameters.maxLevel, readParameters.maxLevel);
    ASSERT_EQ(defaultParameters.maxMedianLengthOfDisplacementDifference,
              readParameters.maxMedianLengthOfDisplacementDifference);
    ASSERT_EQ(defaultParameters.pointsInGrid, readParameters.pointsInGrid);
    ASSERT_EQ(defaultParameters.winSize, readParameters.winSize);
    ASSERT_EQ(defaultParameters.winSizeNCC, readParameters.winSizeNCC);
    ASSERT_EQ(defaultParameters.termCriteria.epsilon, readParameters.termCriteria.epsilon);
    ASSERT_EQ(defaultParameters.termCriteria.maxCount, readParameters.termCriteria.maxCount);
}

TEST(KCF_Parameters, IO)
{
    TrackerKCF::Params parameters;

    parameters.sigma = 0.3f;
    parameters.lambda = 0.02f;
    parameters.interp_factor = 0.08f;
    parameters.output_sigma_factor = 1.0f/ 32.0f;
    parameters.resize=false;
    parameters.max_patch_size=90*90;
    parameters.split_coeff=false;
    parameters.wrap_kernel=true;
    parameters.desc_npca = TrackerKCF::CN;
    parameters.desc_pca = TrackerKCF::GRAY;
    parameters.compress_feature=false;
    parameters.compressed_size=3;
    parameters.pca_learning_rate=0.2f;

    FileStorage fsWriter("parameters.xml", FileStorage::WRITE + FileStorage::MEMORY);
    parameters.write(fsWriter);

    String serializedParameters = fsWriter.releaseAndGetString();

    FileStorage fsReader(serializedParameters, FileStorage::READ + FileStorage::MEMORY);

    TrackerKCF::Params readParameters;
    readParameters.read(fsReader.root());

    ASSERT_DOUBLE_EQ(parameters.sigma, readParameters.sigma);
    ASSERT_DOUBLE_EQ(parameters.lambda, readParameters.lambda);
    ASSERT_DOUBLE_EQ(parameters.interp_factor, readParameters.interp_factor);
    ASSERT_DOUBLE_EQ(parameters.output_sigma_factor, readParameters.output_sigma_factor);
    ASSERT_EQ(parameters.resize, readParameters.resize);
    ASSERT_EQ(parameters.max_patch_size, readParameters.max_patch_size);
    ASSERT_EQ(parameters.split_coeff, readParameters.split_coeff);
    ASSERT_EQ(parameters.wrap_kernel, readParameters.wrap_kernel);
    ASSERT_EQ(parameters.desc_npca, readParameters.desc_npca);
    ASSERT_EQ(parameters.desc_pca, readParameters.desc_pca);
    ASSERT_EQ(parameters.compress_feature, readParameters.compress_feature);
    ASSERT_EQ(parameters.compressed_size, readParameters.compressed_size);
    ASSERT_DOUBLE_EQ(parameters.pca_learning_rate, readParameters.pca_learning_rate);
}

TEST(KCF_Parameters, Default_Value_If_Absent)
{
    TrackerKCF::Params defaultParameters;

    FileStorage fsReader(String("%YAML 1.0"), FileStorage::READ + FileStorage::MEMORY);

    TrackerKCF::Params readParameters;
    readParameters.read(fsReader.root());

    ASSERT_DOUBLE_EQ(defaultParameters.sigma, readParameters.sigma);
    ASSERT_DOUBLE_EQ(defaultParameters.lambda, readParameters.lambda);
    ASSERT_DOUBLE_EQ(defaultParameters.interp_factor, readParameters.interp_factor);
    ASSERT_DOUBLE_EQ(defaultParameters.output_sigma_factor, readParameters.output_sigma_factor);
    ASSERT_EQ(defaultParameters.resize, readParameters.resize);
    ASSERT_EQ(defaultParameters.max_patch_size, readParameters.max_patch_size);
    ASSERT_EQ(defaultParameters.split_coeff, readParameters.split_coeff);
    ASSERT_EQ(defaultParameters.wrap_kernel, readParameters.wrap_kernel);
    ASSERT_EQ(defaultParameters.desc_npca, readParameters.desc_npca);
    ASSERT_EQ(defaultParameters.desc_pca, readParameters.desc_pca);
    ASSERT_EQ(defaultParameters.compress_feature, readParameters.compress_feature);
    ASSERT_EQ(defaultParameters.compressed_size, readParameters.compressed_size);
    ASSERT_DOUBLE_EQ(defaultParameters.pca_learning_rate, readParameters.pca_learning_rate);
}
