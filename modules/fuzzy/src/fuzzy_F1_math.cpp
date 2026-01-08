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

void ft::FT12D_components(InputArray matrix, InputArray kernel, OutputArray components)
{
    Mat c00, c10, c01;

    FT12D_polynomial(matrix, kernel, c00, c10, c01, components);
}

void ft::FT12D_polynomial(InputArray matrix, InputArray kernel, OutputArray c00, OutputArray c10, OutputArray c01, OutputArray components, InputArray mask)
{
    CV_Assert(matrix.channels() == 1 && kernel.channels() == 1);

    Mat inputMask;

    if (mask.getMat().empty())
    {
        inputMask = Mat::ones(matrix.size(), CV_8U);
    }
    else
    {
        CV_Assert(mask.channels() == 1);

        inputMask = mask.getMat();
    }

    int radiusX = (kernel.cols() - 1) / 2;
    int radiusY = (kernel.rows() - 1) / 2;
    int An = matrix.cols() / radiusX + 1;
    int Bn = matrix.rows() / radiusY + 1;

    Mat matrixPadded, maskPadded;
    copyMakeBorder(matrix, matrixPadded, radiusY, kernel.rows(), radiusX, kernel.cols(), BORDER_ISOLATED, Scalar(0));
    copyMakeBorder(inputMask, maskPadded, radiusY, kernel.rows(), radiusX, kernel.cols(), BORDER_ISOLATED, Scalar(0));

    c00.create(Bn, An, CV_32F);
    c10.create(Bn, An, CV_32F);
    c01.create(Bn, An, CV_32F);
    components.create(Bn * kernel.rows(), An * kernel.cols(), CV_32F);

    Mat c00Mat = c00.getMat();
    Mat c10Mat = c10.getMat();
    Mat c01Mat = c01.getMat();
    Mat componentsMat = components.getMat();

    Mat vecX, vecY;
    FT12D_createPolynomMatrixVertical(radiusX, vecX, 1);
    FT12D_createPolynomMatrixHorizontal(radiusY, vecY, 1);

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

            Mat numerator00, numerator10, numerator01;
            multiply(roiImage, kernelMasked, numerator00, 1, CV_32F);
            multiply(numerator00, vecX, numerator10, 1, CV_32F);
            multiply(numerator00, vecY, numerator01, 1, CV_32F);

            Mat denominator00, denominator10, denominator01;
            denominator00 = kernelMasked;
            multiply(vecX.mul(vecX), kernelMasked, denominator10, 1, CV_32F);
            multiply(vecY.mul(vecY), kernelMasked, denominator01, 1, CV_32F);

            Scalar c00sum, c10sum, c01sum;
            divide(sum(numerator00), sum(denominator00), c00sum, 1, CV_32F);
            divide(sum(numerator10), sum(denominator10), c10sum, 1, CV_32F);
            divide(sum(numerator01), sum(denominator01), c01sum, 1, CV_32F);

            c00Mat.row(o).col(i) = c00sum;
            c10Mat.row(o).col(i) = c10sum;
            c01Mat.row(o).col(i) = c01sum;

            Mat vecXMasked, vecYMasked;
            vecX.copyTo(vecXMasked, roiMask);
            vecY.copyTo(vecYMasked, roiMask);

            Mat updatedC10, updatedC01;
            multiply(c10sum, vecXMasked, updatedC10, 1, CV_32F);
            multiply(c01sum, vecYMasked, updatedC01, 1, CV_32F);

            Mat component(componentsMat, Rect(i * kernelMasked.cols, o * kernelMasked.rows, kernelMasked.cols, kernelMasked.rows));
            add(updatedC01, updatedC10, component);
            add(component, c00sum, component);
        }
    }
}

void ft::FT12D_createPolynomMatrixVertical(int radius, OutputArray matrix, const int chn)
{
    int dimension = radius * 2 + 1;

    std::vector<Mat> channels;
    Mat oneChannel(dimension, dimension, CV_16SC1, Scalar(0));

    for (int i = 0; i < radius; i++)
    {
        oneChannel.col(i) = i - radius;
        oneChannel.col(dimension - 1 - i) = radius - i;
    }

    for (int i = 0; i < chn; i++)
    {
        channels.push_back(oneChannel);
    }

    merge(channels, matrix);
}

void ft::FT12D_createPolynomMatrixHorizontal(int radius, OutputArray matrix, const int chn)
{
    int dimension = radius * 2 + 1;

    std::vector<Mat> channels;
    Mat oneChannel(dimension, dimension, CV_16SC1, Scalar(0));

    for (int i = 0; i < radius; i++)
    {
        oneChannel.row(i) = i - radius;
        oneChannel.row(dimension - 1 - i) = radius - i;
    }

    for (int i = 0; i < chn; i++)
    {
        channels.push_back(oneChannel);
    }

    merge(channels, matrix);
}

void ft::FT12D_inverseFT(InputArray components, InputArray kernel, OutputArray output, int width, int height)
{
    CV_Assert(components.channels() == 1 && kernel.channels() == 1);

    Mat componentsMat = components.getMat();

    int radiusX = (kernel.cols() - 1) / 2;
    int radiusY = (kernel.rows() - 1) / 2;
    int outputWidthPadded = radiusX + width + kernel.cols();
    int outputHeightPadded = radiusY + height + kernel.rows();

    output.create(height, width, CV_32F);

    Mat outputZeroes(outputHeightPadded, outputWidthPadded, CV_32F, Scalar(0));

    for (int i = 0; i < componentsMat.cols / kernel.cols(); i++)
    {
        for (int o = 0; o < componentsMat.rows / kernel.rows(); o++)
        {
            int centerX = (i * radiusX) + radiusX;
            int centerY = (o * radiusY) + radiusY;
            Rect area(centerX - radiusX, centerY - radiusY, kernel.cols(), kernel.rows());

            Mat component(componentsMat, Rect(i * kernel.cols(), o * kernel.rows(), kernel.cols(), kernel.rows()));

            Mat inverse;
            multiply(kernel, component, inverse, 1, CV_32F);

            Mat roiOutput(outputZeroes, area);
            add(roiOutput, inverse, roiOutput);
        }
    }

    outputZeroes(Rect(radiusX, radiusY, width, height)).copyTo(output);
}

void ft::FT12D_process(InputArray matrix, InputArray kernel, OutputArray output, InputArray mask)
{
    CV_Assert(matrix.channels() == kernel.channels());

    Mat inputMask;

    if (mask.getMat().empty())
    {
        inputMask = Mat::ones(matrix.size(), CV_8U);
    }
    else
    {
        CV_Assert(mask.channels() == 1);

        inputMask = mask.getMat();
    }

    Mat matrixPadded;
    Mat maskPadded;

    int radiusX = (kernel.cols() - 1) / 2;
    int radiusY = (kernel.rows() - 1) / 2;
    int An = matrix.cols() / radiusX + 1;
    int Bn = matrix.rows() / radiusY + 1;
    int outputWidthPadded = radiusX + matrix.cols() + kernel.cols();
    int outputHeightPadded = radiusY + matrix.rows() + kernel.rows();

    output.create(matrix.size(), CV_MAKETYPE(CV_32F, matrix.channels()));

    Mat outputZeroes(outputHeightPadded, outputWidthPadded, output.type(), Scalar(0));

    copyMakeBorder(matrix, matrixPadded, radiusY, kernel.rows(), radiusX, kernel.cols(), BORDER_CONSTANT, Scalar(0));
    copyMakeBorder(inputMask, maskPadded, radiusY, kernel.rows(), radiusX, kernel.cols(), BORDER_CONSTANT, Scalar(0));

    Mat vecX;
    Mat vecY;

    ft::FT12D_createPolynomMatrixVertical(radiusX, vecX, matrix.channels());
    ft::FT12D_createPolynomMatrixHorizontal(radiusY, vecY, matrix.channels());

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

            Mat numerator00, numerator10, numerator01;
            multiply(roiImage, kernelMasked, numerator00, 1, CV_32F);
            multiply(numerator00, vecX, numerator10, 1, CV_32F);
            multiply(numerator00, vecY, numerator01, 1, CV_32F);

            Mat denominator00, denominator10, denominator01;
            denominator00 = kernelMasked;
            multiply(vecX.mul(vecX), kernelMasked, denominator10, 1, CV_32F);
            multiply(vecY.mul(vecY), kernelMasked, denominator01, 1, CV_32F);

            Scalar c00, c10, c01;
            divide(sum(numerator00), sum(denominator00), c00, 1, CV_32F);
            divide(sum(numerator10), sum(denominator10), c10, 1, CV_32F);
            divide(sum(numerator01), sum(denominator01), c01, 1, CV_32F);

            Mat component, updatedC10, updatedC01;

            multiply(c10, vecX, updatedC10, 1, CV_32F);
            multiply(c01, vecY, updatedC01, 1, CV_32F);

            add(updatedC01, updatedC10, component);
            add(component, c00, component);

            Mat inverse;
            multiply(kernel, component, inverse, 1, CV_32F);

            Mat roiOutput(outputZeroes, area);
            add(roiOutput, inverse, roiOutput);
        }
    }

    outputZeroes(Rect(radiusX, radiusY, matrix.cols(), matrix.rows())).copyTo(output);
}
