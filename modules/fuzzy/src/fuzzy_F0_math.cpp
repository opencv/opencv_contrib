/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2015, University of Ostrava, Institute for Research and Applications of Fuzzy Modeling,
// Pavel Vlasanek, all rights reserved. Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

using namespace cv;

void ft::FT02D_components(InputArray matrix, InputArray kernel, OutputArray components, InputArray mask)
{
    CV_Assert(matrix.channels() == kernel.channels() && mask.channels() == 1);

    int radiusX = (kernel.cols() - 1) / 2;
    int radiusY = (kernel.rows() - 1) / 2;
    int An = matrix.cols() / radiusX + 1;
    int Bn = matrix.rows() / radiusY + 1;

    Mat matrixPadded;
    Mat maskPadded;

    copyMakeBorder(matrix, matrixPadded, radiusY, kernel.rows(), radiusX, kernel.cols(), BORDER_CONSTANT, Scalar(0));
    copyMakeBorder(mask, maskPadded, radiusY, kernel.rows(), radiusX, kernel.cols(), BORDER_CONSTANT, Scalar(0));

    components.create(Bn, An, CV_MAKETYPE(CV_32F, matrix.channels()));

    Mat componentsMat = components.getMat();

    for (int i = 0; i < An; i++)
    {
        for (int o = 0; o < Bn; o++)
        {
            int centerX = (i * radiusX) + radiusX;
            int centerY = (o * radiusY) + radiusY;
            Rect area(centerX - radiusX, centerY - radiusY, kernel.cols(), kernel.rows());

            Mat roiImage(matrixPadded, area);
            Mat roiMask(maskPadded, area);
            Mat kernelMasked;

            kernel.copyTo(kernelMasked, roiMask);

            Mat numerator;
            multiply(roiImage, kernelMasked, numerator, 1, CV_32F);

            Scalar value;
            divide(sum(numerator), sum(kernelMasked), value, 1, CV_32F);

            componentsMat.row(o).col(i).setTo(value);
        }
    }
}

void ft::FT02D_components(InputArray matrix, InputArray kernel, OutputArray components)
{
    Mat mask = Mat::ones(matrix.size(), CV_8U);

    ft::FT02D_components(matrix, kernel, components, mask);
}

void ft::FT02D_inverseFT(InputArray components, InputArray kernel, OutputArray output, int width, int height)
{
    CV_Assert(components.channels() == 1 && kernel.channels() == 1);

    Mat componentsMat = components.getMat();

    int radiusX = (kernel.cols() - 1) / 2;
    int radiusY = (kernel.rows() - 1) / 2;
    int outputWidthPadded = radiusX + width + kernel.cols();
    int outputHeightPadded = radiusY + height + kernel.rows();

    output.create(height, width, CV_32F);

    Mat outputZeroes(outputHeightPadded, outputWidthPadded, CV_32F, Scalar(0));

    for (int i = 0; i < componentsMat.cols; i++)
    {
        for (int o = 0; o < componentsMat.rows; o++)
        {
            int centerX = (i * radiusX) + radiusX;
            int centerY = (o * radiusY) + radiusY;
            Rect area(centerX - radiusX, centerY - radiusY, kernel.cols(), kernel.rows());

            float component = componentsMat.at<float>(o, i);

            Mat inverse;
            multiply(kernel, component, inverse, 1, CV_32F);

            Mat roiOutput(outputZeroes, area);
            add(roiOutput, inverse, roiOutput);
        }
    }

    outputZeroes(Rect(radiusX, radiusY, width, height)).copyTo(output);
}

void ft::FT02D_process(InputArray matrix, InputArray kernel, OutputArray output)
{
    Mat mask = Mat::ones(matrix.size(), CV_8U);

    ft::FT02D_process(matrix, kernel, output, mask);
}

void ft::FT02D_process(InputArray matrix, InputArray kernel, OutputArray output, InputArray mask)
{
    CV_Assert(matrix.channels() == kernel.channels() && mask.channels() == 1);

    int radiusX = (kernel.cols() - 1) / 2;
    int radiusY = (kernel.rows() - 1) / 2;
    int An = matrix.cols() / radiusX + 1;
    int Bn = matrix.rows() / radiusY + 1;
    int outputWidthPadded = radiusX + matrix.cols() + kernel.cols();
    int outputHeightPadded = radiusY + matrix.rows() + kernel.rows();

    Mat matrixPadded;
    Mat maskPadded;

    output.create(matrix.size(), CV_MAKETYPE(CV_32F, matrix.channels()));

    Mat outputZeroes(outputHeightPadded, outputWidthPadded, output.type(), Scalar(0));

    copyMakeBorder(matrix, matrixPadded, radiusY, kernel.rows(), radiusX, kernel.cols(), BORDER_CONSTANT, Scalar(0));
    copyMakeBorder(mask, maskPadded, radiusY, kernel.rows(), radiusX, kernel.cols(), BORDER_CONSTANT, Scalar(0));

    for (int i = 0; i < An; i++)
    {
        for (int o = 0; o < Bn; o++)
        {
            int centerX = (i * radiusX) + radiusX;
            int centerY = (o * radiusY) + radiusY;
            Rect area(centerX - radiusX, centerY - radiusY, kernel.cols(), kernel.rows());

            Mat roiMatrix(matrixPadded, area);
            Mat roiMask(maskPadded, area);
            Mat kernelMasked;

            kernel.copyTo(kernelMasked, roiMask);

            Mat numerator;
            multiply(roiMatrix, kernelMasked, numerator, 1, CV_32F);

            Scalar component;
            divide(sum(numerator), sum(kernelMasked), component, 1, CV_32F);

            Mat inverse;
            multiply(kernel, component, inverse, 1, CV_32F);

            Mat roiOutput(outputZeroes, area);
            add(roiOutput, inverse, roiOutput);
        }
    }

    outputZeroes(Rect(radiusX, radiusY, matrix.cols(), matrix.rows())).copyTo(output);
}

int ft::FT02D_iteration(InputArray matrix, InputArray kernel, OutputArray output, InputArray mask, OutputArray maskOutput, bool firstStop)
{
    CV_Assert(matrix.channels() == kernel.channels() && mask.channels() == 1);

    int radiusX = (kernel.cols() - 1) / 2;
    int radiusY = (kernel.rows() - 1) / 2;
    int An = matrix.cols() / radiusX + 1;
    int Bn = matrix.rows() / radiusY + 1;
    int outputWidthPadded = radiusX + matrix.cols() + kernel.cols();
    int outputHeightPadded = radiusY + matrix.rows() + kernel.rows();
    int undefinedComponents = 0;

    output.create(matrix.size(), CV_MAKETYPE(CV_32F, matrix.channels()));
    output.setTo(0);

    if (maskOutput.needed())
    {
        maskOutput.create(mask.rows(), mask.cols(), CV_8UC1);
        maskOutput.setTo(1);
    }

    Mat matrixOutputMat = Mat::zeros(outputHeightPadded, outputWidthPadded, CV_MAKETYPE(CV_32F, matrix.channels()));
    Mat maskOutputMat = Mat::ones(outputHeightPadded, outputWidthPadded, CV_8UC1);

    Mat matrixPadded;
    Mat maskPadded;

    copyMakeBorder(matrix, matrixPadded, radiusY, kernel.rows(), radiusX, kernel.cols(), BORDER_CONSTANT, Scalar(0));
    copyMakeBorder(mask, maskPadded, radiusY, kernel.rows(), radiusX, kernel.cols(), BORDER_CONSTANT, Scalar(0));

    for (int i = 0; i < An; i++)
    {
        for (int o = 0; o < Bn; o++)
        {
            int centerX = (i * radiusX) + radiusX;
            int centerY = (o * radiusY) + radiusY;
            Rect area(centerX - radiusX, centerY - radiusY, kernel.cols(), kernel.rows());

            Mat roiMatrix(matrixPadded, area);
            Mat roiMask(maskPadded, area);
            Mat kernelMasked;

            kernel.copyTo(kernelMasked, roiMask);

            Mat numerator;
            multiply(roiMatrix, kernelMasked, numerator, 1, CV_32F);

            Scalar denominator = sum(kernelMasked);

            if (denominator[0] == 0)
            {
                if (firstStop)
                {
                    matrixOutputMat = matrixPadded(Rect(radiusX, radiusY, matrix.cols(), matrix.rows()));
                    maskOutputMat = maskPadded(Rect(radiusX, radiusY, matrix.cols(), matrix.rows()));

                    return -1;
                }
                else
                {
                    undefinedComponents++;

                    Mat roiMaskOutput(maskOutputMat, Rect(centerX - radiusX + 1, centerY - radiusY + 1, kernel.cols() - 2, kernel.rows() - 2));
                    roiMaskOutput.setTo(0);

                    continue;
                }
            }

            Scalar component;
            divide(sum(numerator), denominator, component, 1, CV_32F);

            Mat inverse;
            multiply(kernel, component, inverse, 1, CV_32F);

            Mat roiMatrixOutput(matrixOutputMat, area);
            add(roiMatrixOutput, inverse, roiMatrixOutput);
        }
    }

    matrixOutputMat(Rect(radiusX, radiusY, matrix.cols(), matrix.rows())).copyTo(output);

    if (maskOutput.needed())
    {
        maskOutputMat(Rect(radiusX, radiusY, matrix.cols(), matrix.rows())).copyTo(maskOutput);
    }

    return undefinedComponents;
}
