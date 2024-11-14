/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

static void runMSER(InputArray _src, std::vector<std::vector<Point>>& contours, std::vector<cv::Rect>& boundingBoxes,
                    std::vector<ContourData>& contourData,
                    bool useBoundingBoxes = true,
                    bool useContourData = true,
                    unsigned int numNeighbors = 4,
                    unsigned int delta = 2,
                    unsigned int minArea = 30,
                    unsigned int maxArea = 14400,
                    float        maxVariation = 0.15f,
                    float        minDiversity = 0.2f)
{
    INITIALIZATION_CHECK;

    CV_Assert(!_src.empty() && _src.type() == CV_8UC1);
    CV_Assert(_src.cols() > 50);
    CV_Assert(_src.rows() > 5);

    Mat src = _src.getMat();

    CV_Assert(numNeighbors == 4 || numNeighbors == 8);
    bool useNN4 = (numNeighbors == 4);

    bool usePointsArray = !useNN4;

    void *mserHandle;

    bool isInitOk = false;
    if (useNN4)
    {
        isInitOk = fcvMserInit(src.cols, src.rows, delta, minArea, maxArea, maxVariation, minDiversity, &mserHandle);
    }
    else
    {
        isInitOk = fcvMserNN8Init(src.cols, src.rows, delta, minArea, maxArea, maxVariation, minDiversity, &mserHandle);
    }

    if (!isInitOk)
    {
        CV_Error(cv::Error::StsInternal, "Failed to initialize MSER");
    }

    //bufSize for pts and bboxes
    const unsigned int maxContours = 16384;
    unsigned int numContours;
    std::vector<uint32_t> numPointsInContour(maxContours);

    std::vector<uint16_t> rectArray;
    rectArray.resize(4 * maxContours); // xMin, xMax, yMax, yMin

    unsigned int pointsArraySize = src.total() * 30; // Recommended typical size
    std::vector<uint16_t> pointsArray;
    std::vector<uint32_t> contourStartingPoints;
    uint32_t pathArraySize = src.total() * 4; // Recommended size
    std::vector<uint16_t> pathArray;
    if (usePointsArray)
    {
        pointsArray.resize(pointsArraySize);
    }
    else
    {
        contourStartingPoints.resize(maxContours);
        pathArray.resize(pathArraySize);
    }

    std::vector<uint32_t> contourVariation(maxContours), contourNodeId(maxContours), contourNodeCounter(maxContours);
    std::vector<int8_t> contourPolarity(maxContours);

    int mserRetcode = -1;
    if (useNN4)
    {
        mserRetcode = fcvMserExtu8_v3(mserHandle, src.data, src.cols, src.rows, src.step,
                                      maxContours, &numContours,
                                      rectArray.data(),
                                      contourStartingPoints.data(),
                                      numPointsInContour.data(),
                                      pathArraySize, pathArray.data(),
                                      contourVariation.data(), contourPolarity.data(), contourNodeId.data(), contourNodeCounter.data());
        CV_LOG_INFO(NULL, "fcvMserExtu8_v3");
    }
    else
    {
        if (useContourData)
        {
            mserRetcode = fcvMserExtNN8u8(mserHandle, src.data, src.cols, src.rows, src.step,
                                          maxContours, &numContours,
                                          rectArray.data(),
                                          numPointsInContour.data(), pointsArraySize, pointsArray.data(),
                                          contourVariation.data(), contourPolarity.data(), contourNodeId.data(), contourNodeCounter.data());
            CV_LOG_INFO(NULL, "fcvMserExtNN8u8");
        }
        else
        {
            mserRetcode = fcvMserNN8u8(mserHandle, src.data, src.cols, src.rows, src.step,
                                       maxContours, &numContours,
                                       rectArray.data(),
                                       numPointsInContour.data(), pointsArraySize, pointsArray.data());
            CV_LOG_INFO(NULL, "fcvMserNN8u8");
        }
    }

    if (mserRetcode != 1)
    {
        CV_Error(cv::Error::StsInternal, "Failed to run MSER");
    }

    contours.clear();
    contours.reserve(numContours);
    if (useBoundingBoxes)
    {
        boundingBoxes.clear();
        boundingBoxes.reserve(numContours);
    }
    if (useContourData)
    {
        contourData.clear();
        contourData.reserve(numContours);
    }
    int ptCtr = 0;
    for (uint32_t i = 0; i < numContours; i++)
    {
        std::vector<Point> contour;
        contour.reserve(numPointsInContour[i]);
        for (uint32_t j = 0; j < numPointsInContour[i]; j++)
        {
            Point pt;
            if (usePointsArray)
            {
                uint32_t idx = (ptCtr + j) * 2;
                pt = Point {pointsArray[idx + 0], pointsArray[idx + 1]};
            }
            else
            {
                uint32_t idx = contourStartingPoints[i] + j * 2;
                pt = Point {pathArray[idx + 0], pathArray[idx + 1]};
            }
            contour.push_back(pt);
        }
        contours.push_back(contour);
        ptCtr += numPointsInContour[i];

        if (useBoundingBoxes)
        {
            uint16_t xMin = rectArray[i * 4 + 0];
            uint16_t xMax = rectArray[i * 4 + 1];
            uint16_t yMax = rectArray[i * 4 + 2];
            uint16_t yMin = rectArray[i * 4 + 3];
            // +1 is because max limit in cv::Rect() is exclusive
            cv::Rect bbox(Point {xMin, yMin},
                          Point {xMax + 1, yMax + 1});
            boundingBoxes.push_back(bbox);
        }

        if (useContourData)
        {
            ContourData data;
            data.variation   = contourVariation[i];
            data.polarity    = contourPolarity[i];
            data.nodeId      = contourNodeId[i];
            data.nodeCounter = contourNodeCounter[i];
            contourData.push_back(data);
        }
    }

    fcvMserRelease(mserHandle);
}

void MSER(InputArray _src, std::vector<std::vector<Point>> &contours,
          unsigned int numNeighbors, unsigned int delta, unsigned int minArea, unsigned int maxArea, float maxVariation, float minDiversity)
{
    std::vector<cv::Rect> boundingBoxes;
    std::vector<ContourData> contourData;
    runMSER(_src, contours, boundingBoxes, contourData, false, false, numNeighbors,
            delta, minArea, maxArea, maxVariation, minDiversity);
}

void MSER(InputArray _src, std::vector<std::vector<Point>>& contours, std::vector<cv::Rect>& boundingBoxes,
          unsigned int numNeighbors, unsigned int delta, unsigned int minArea, unsigned int maxArea, float maxVariation, float minDiversity)
{
    std::vector<ContourData> contourData;
    runMSER(_src, contours, boundingBoxes, contourData, true, false, numNeighbors,
            delta, minArea, maxArea, maxVariation, minDiversity);
}

void MSER(InputArray _src, std::vector<std::vector<Point>>& contours, std::vector<cv::Rect>& boundingBoxes, std::vector<ContourData>& contourData,
          unsigned int numNeighbors, unsigned int delta, unsigned int minArea, unsigned int maxArea, float maxVariation, float minDiversity)
{
    runMSER(_src, contours, boundingBoxes, contourData, true, true, numNeighbors,
            delta, minArea, maxArea, maxVariation, minDiversity);
}

} // fastcv::
} // cv::
