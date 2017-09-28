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

void ft::FT02D_FL_process(InputArray matrix, const int radius, OutputArray output)
{
    CV_Assert(matrix.channels() == 3);

    int borderPadding = 2 * radius + 1;
    Mat imagePadded;

    copyMakeBorder(matrix, imagePadded, radius, borderPadding, radius, borderPadding, BORDER_CONSTANT, Scalar(0));

    Mat channel[3];
    split(imagePadded, channel);

    uchar *im_r = channel[2].data;
    uchar *im_g = channel[1].data;
    uchar *im_b = channel[0].data;

    int width = imagePadded.cols;
    int height = imagePadded.rows;
    int n_width = width / radius + 1;
    int n_height = height / radius + 1;

    std::vector<uchar> c_r(n_width * n_height);
    std::vector<uchar> c_g(n_width * n_height);
    std::vector<uchar> c_b(n_width * n_height);

    int sum_r, sum_g, sum_b, num, c_wei;
    int c_pos, pos, pos2, wy;
    int cy = 0;
    float num_f;

    std::vector<int> wei(radius + 1);

    for (int i = 0; i <= radius; i++)
    {
        wei[i] = radius - i;
    }

    for (int y = radius; y < height - radius; y += radius)
    {
        c_pos = cy;

        for (int x = radius; x < width - radius; x += radius)
        {
            num = sum_r = sum_g = sum_b = 0;

            for (int y1 = y - radius; y1 <= y + radius; y1++)
            {
                pos = y1 * width;
                wy = wei[abs(y1 - y)];

                for (int x1 = x - radius; x1 <= x + radius; x1++)
                {
                    c_wei = wei[abs(x1 - x)] * wy;
                    pos2 = pos + x1;
                    sum_r += im_r[pos2] * c_wei;
                    sum_g += im_g[pos2] * c_wei;
                    sum_b += im_b[pos2] * c_wei;
                    num += c_wei;
                }
            }

            num_f = 1.0f / (float)num;

            c_r[c_pos] = (uchar)cvRound(sum_r * num_f);
            c_g[c_pos] = (uchar)cvRound(sum_g * num_f);
            c_b[c_pos] = (uchar)cvRound(sum_b * num_f);

            c_pos++;
        }

        cy += n_width;
    }

    int p1, p2, p3, p4, yw, w1, w2, w3, w4, lx, ly, lx1, ly1, pos_iFT;
    float num_iFT;

    int output_height = matrix.rows();
    int output_width = matrix.cols();

    uchar *img_r = new uchar[output_height * output_width];
    uchar *img_g = new uchar[output_height * output_width];
    uchar *img_b = new uchar[output_height * output_width];

    for (int y = 0; y < output_height; y++)
    {
        ly1 = (y % radius);
        ly = radius - ly1;
        yw = y / radius * n_width;
        pos_iFT = y * output_width;

        for (int x = 0; x < output_width; x++)
        {
            lx1 = (x % radius);
            lx = radius - lx1;

            p1 = x / radius + yw;
            p2 = p1 + 1;
            p3 = p1 + n_width;
            p4 = p3 + 1;

            w1 = lx * ly;
            w2 = lx1 * ly;
            w3 = lx * ly1;
            w4 = lx1 * ly1;

            num_iFT = 1.0f / (float)(w1 + w2 + w3 + w4);

            img_r[pos_iFT] = (uchar)((c_r[p1] * w1 + c_r[p2] * w2 + c_r[p3] * w3 + c_r[p4] * w4) * num_iFT);
            img_g[pos_iFT] = (uchar)((c_g[p1] * w1 + c_g[p2] * w2 + c_g[p3] * w3 + c_g[p4] * w4) * num_iFT);
            img_b[pos_iFT] = (uchar)((c_b[p1] * w1 + c_b[p2] * w2 + c_b[p3] * w3 + c_b[p4] * w4) * num_iFT);

            pos_iFT++;
        }
    }

    Mat compR(output_height, output_width, CV_8UC1, img_r);
    Mat compG(output_height, output_width, CV_8UC1, img_g);
    Mat compB(output_height, output_width, CV_8UC1, img_b);

    std::vector<Mat> oComp;

    oComp.push_back(compB);
    oComp.push_back(compG);
    oComp.push_back(compR);

    merge(oComp, output);
}

void ft::FT02D_FL_process_float(InputArray matrix, const int radius, OutputArray output)
{
    CV_Assert(matrix.channels() == 3);

    int borderPadding = 2 * radius + 1;
    Mat imagePadded;

    copyMakeBorder(matrix, imagePadded, radius, borderPadding, radius, borderPadding, BORDER_CONSTANT, Scalar(0));

    Mat channel[3];
    split(imagePadded, channel);

    uchar *im_r = channel[2].data;
    uchar *im_g = channel[1].data;
    uchar *im_b = channel[0].data;

    int width = imagePadded.cols;
    int height = imagePadded.rows;
    int n_width = width / radius + 1;
    int n_height = height / radius + 1;

    std::vector<float> c_r(n_width * n_height);
    std::vector<float> c_g(n_width * n_height);
    std::vector<float> c_b(n_width * n_height);

    int sum_r, sum_g, sum_b, num, c_wei;
    int c_pos, pos, pos2, wy;
    int cy = 0;
    float num_f;

    std::vector<int> wei(radius + 1);

    for (int i = 0; i <= radius; i++)
    {
        wei[i] = radius - i;
    }

    for (int y = radius; y < height - radius; y += radius)
    {
        c_pos = cy;

        for (int x = radius; x < width - radius; x += radius)
        {
            num = sum_r = sum_g = sum_b = 0;

            for (int y1 = y - radius; y1 <= y + radius; y1++)
            {
                pos = y1 * width;
                wy = wei[abs(y1 - y)];

                for (int x1 = x - radius; x1 <= x + radius; x1++)
                {
                    c_wei = wei[abs(x1 - x)] * wy;
                    pos2 = pos + x1;
                    sum_r += im_r[pos2] * c_wei;
                    sum_g += im_g[pos2] * c_wei;
                    sum_b += im_b[pos2] * c_wei;
                    num += c_wei;
                }
            }

            num_f = 1.0f / (float)num;

            c_r[c_pos] = sum_r * num_f;
            c_g[c_pos] = sum_g * num_f;
            c_b[c_pos] = sum_b * num_f;

            c_pos++;
        }

        cy += n_width;
    }

    int p1, p2, p3, p4, yw, w1, w2, w3, w4, lx, ly, lx1, ly1, pos_iFT;
    float num_iFT;

    int output_height = matrix.rows();
    int output_width = matrix.cols();

    float *img_r = new float[output_height * output_width];
    float *img_g = new float[output_height * output_width];
    float *img_b = new float[output_height * output_width];

    for (int y = 0; y < output_height; y++)
    {
        ly1 = (y % radius);
        ly = radius - ly1;
        yw = y / radius * n_width;
        pos_iFT = y * output_width;

        for (int x = 0; x < output_width; x++)
        {
            lx1 = (x % radius);
            lx = radius - lx1;

            p1 = x / radius + yw;
            p2 = p1 + 1;
            p3 = p1 + n_width;
            p4 = p3 + 1;

            w1 = lx * ly;
            w2 = lx1 * ly;
            w3 = lx * ly1;
            w4 = lx1 * ly1;

            num_iFT = 1.0f / (float)(w1 + w2 + w3 + w4);

            img_r[pos_iFT] = (c_r[p1] * w1 + c_r[p2] * w2 + c_r[p3] * w3 + c_r[p4] * w4) * num_iFT;
            img_g[pos_iFT] = (c_g[p1] * w1 + c_g[p2] * w2 + c_g[p3] * w3 + c_g[p4] * w4) * num_iFT;
            img_b[pos_iFT] = (c_b[p1] * w1 + c_b[p2] * w2 + c_b[p3] * w3 + c_b[p4] * w4) * num_iFT;

            pos_iFT++;
        }
    }

    Mat compR(output_height, output_width, CV_32FC1, img_r);
    Mat compG(output_height, output_width, CV_32FC1, img_g);
    Mat compB(output_height, output_width, CV_32FC1, img_b);

    std::vector<Mat> oComp;

    oComp.push_back(compB);
    oComp.push_back(compG);
    oComp.push_back(compR);

    merge(oComp, output);
}

void ft::FT02D_components(InputArray matrix, InputArray kernel, OutputArray components, InputArray mask)
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

    int radiusX = (kernel.cols() - 1) / 2;
    int radiusY = (kernel.rows() - 1) / 2;
    int An = matrix.cols() / radiusX + 1;
    int Bn = matrix.rows() / radiusY + 1;

    Mat matrixPadded;
    Mat maskPadded;

    copyMakeBorder(matrix, matrixPadded, radiusY, kernel.rows(), radiusX, kernel.cols(), BORDER_CONSTANT, Scalar(0));
    copyMakeBorder(inputMask, maskPadded, radiusY, kernel.rows(), radiusX, kernel.cols(), BORDER_CONSTANT, Scalar(0));

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

void ft::FT02D_process(InputArray matrix, InputArray kernel, OutputArray output, InputArray mask)
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
    copyMakeBorder(inputMask, maskPadded, radiusY, kernel.rows(), radiusX, kernel.cols(), BORDER_CONSTANT, Scalar(0));

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
