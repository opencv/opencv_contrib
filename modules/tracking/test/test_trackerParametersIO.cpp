// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/tracking.hpp"

using namespace cv;

TEST(MEDIAN_FLOW_Parameters_IO, MEDIAN_FLOW)
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


TEST(MEDIAN_FLOW_Parameters_IO_Default_Value_If_Absent, MEDIAN_FLOW)
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
