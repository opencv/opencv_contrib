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
// Copyright (C) 2017, IBM Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//      Marc Fiammante marc.fiammante@fr.ibm.com
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
//   * The name of OpenCV Foundation or contributors may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "brightedges.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <signal.h>
namespace cv
{

	bool isPixelMinimum(Mat &edge, int row, int col, int contrast) {
		int count = 0;
		unsigned char *input = (unsigned char*)(edge.data);
		int pixel = input[edge.cols * row + col] + contrast - 1;
		int m2 = input[edge.cols * (row - 2) + (col - 2)];
		int m1 = input[edge.cols * (row - 1) + (col - 1)];
		int p1 = input[edge.cols * (row + 1) + (col + 1)];
		int p2 = input[edge.cols * (row + 2) + (col + 2)];
		if ((pixel <= m1) && (pixel <= p1) && (pixel < (m1 + m2) / 2) && (pixel < (p1 + p2) / 2)) count++; // Local minimum diagonal
		m2 = input[edge.cols * (row - 2) + (col)];
		m1 = input[edge.cols * (row - 1) + (col)];
		p1 = input[edge.cols * (row + 1) + (col)];
		p2 = input[edge.cols * (row + 2) + (col)];
		if ((pixel <= m1) && (pixel <= p1) && (pixel < (m1 + m2) / 2) && (pixel < (p1 + p2) / 2)) count++; // Local minimum vertical
		m2 = input[edge.cols * (row - 2) + (col + 2)];
		m1 = input[edge.cols * (row - 1) + (col + 1)];
		p1 = input[edge.cols * (row + 1) + (col - 1)];
		p2 = input[edge.cols * (row + 2) + (col - 2)];
		if ((pixel <= m1) && (pixel <= p1) && (pixel < (m1 + m2) / 2) && (pixel < (p1 + p2) / 2)) count++; // Local minimum other diagonal
		m2 = input[edge.cols * (row)+(col + 2)];
		m1 = input[edge.cols * (row)+(col + 1)];
		p1 = input[edge.cols * (row)+(col - 1)];
		p2 = input[edge.cols * (row)+(col - 2)];

		if ((pixel <= m1) && (pixel <= p1) && (pixel < (m1 + m2) / 2) && (pixel < (p1 + p2) / 2)) count++; // Local minimum horizontal

		if (count > 1) return true; // Avoid corners of black zones
		return false;
	}
	int correctPixel(Mat &iedge, int row, int col) {
		unsigned char *input = (unsigned char*)(iedge.data);

		int pixel =
			input[iedge.cols * (row - 1) + (col - 1)] + input[iedge.cols * (row - 1) + col] +
			input[iedge.cols * (row - 1) + (col + 1)] + input[iedge.cols * (row)+(col - 1)] +
			input[iedge.cols * (row)+(col + 1)] + input[iedge.cols * (row + 1) + (col - 1)] +
			input[iedge.cols * (row + 1) + (col)] + input[iedge.cols * (row + 1) + (col + 1)]
			;


		// now check in there is a line around pixel to fill gaps
		// Around Diagonal top left to bottom right
		int weight = 4 * 255;
		int lines = 0;
		int line =
			input[iedge.cols * (row - 1) + (col - 2)] +
			input[iedge.cols * (row - 1) + (col - 1)] +
			input[iedge.cols * (row)+(col + 1)] +
			input[iedge.cols * (row)+(col + 2)];

		if (line == 0) lines += 1;
		line =
			input[iedge.cols * (row)+(col - 2)] +
			input[iedge.cols * (row)+(col - 1)] +
			input[iedge.cols * (row + 1) + (col + 1)] +
			input[iedge.cols * (row + 1) + (col + 2)];
		if (line == 0) lines += 1;

		line =
			input[iedge.cols * (row - 2) + (col)] +
			input[iedge.cols * (row - 1) + (col)] +
			input[iedge.cols * (row + 1) + (col + 1)] +
			input[iedge.cols * (row + 2) + (col + 1)];
		if (line == 0) lines += 1;

		line =
			input[iedge.cols * (row - 2) + (col - 1)] +
			input[iedge.cols * (row - 1) + (col - 1)] +
			input[iedge.cols * (row + 1) + (col)] +
			input[iedge.cols * (row + 2) + (col)];
		if (line == 0) lines += 1;

		line =
			input[iedge.cols * (row - 2) + (col - 2)] +
			input[iedge.cols * (row - 1) + (col - 2)] +
			input[iedge.cols * (row - 2) + (col - 1)] +
			input[iedge.cols * (row - 1) + (col - 1)] +
			input[iedge.cols * (row + 1) + (col + 1)] +
			input[iedge.cols * (row + 1) + (col + 2)] +
			input[iedge.cols * (row + 2) + (col + 1)] +
			input[iedge.cols * (row + 2) + (col + 2)];
		if (line < weight) lines += 1;

		// Near vertical 
		line =
			input[iedge.cols * (row - 2) + (col)] +
			input[iedge.cols * (row - 1) + (col)] +
			input[iedge.cols * (row - 2) + (col - 1)] +
			input[iedge.cols * (row - 2) + (col + 1)] +
			input[iedge.cols * (row + 1) + (col)] +
			input[iedge.cols * (row + 2) + (col)] +
			input[iedge.cols * (row + 2) + (col + 1)] +
			input[iedge.cols * (row + 2) + (col - 1)];
		if (line < weight) lines += 1;
		// Near diagonal top right to bottom left 

		line =
			input[iedge.cols * (row - 2) + (col + 2)] +
			input[iedge.cols * (row - 1) + (col + 1)] +
			input[iedge.cols * (row - 2) + (col - 1)] +
			input[iedge.cols * (row - 1) + (col + 2)] +
			input[iedge.cols * (row + 1) + (col - 1)] +
			input[iedge.cols * (row + 2) + (col - 2)] +
			input[iedge.cols * (row + 2) + (col - 1)] +
			input[iedge.cols * (row + 1) + (col - 2)];
		if (line < weight) lines += 1;

		// Near horizontal 
		line =
			input[iedge.cols * (row)+(col - 2)] +
			input[iedge.cols * (row)+(col - 1)] +
			input[iedge.cols * (row - 1) + (col - 2)] +
			input[iedge.cols * (row + 1) + (col - 2)] +
			input[iedge.cols * (row)+(col + 1)] +
			input[iedge.cols * (row)+(col + 2)] +
			input[iedge.cols * (row + 1) + (col + 2)] +
			input[iedge.cols * (row - 1) + (col + 2)];
		if (line < weight) lines += 1;

		if (line == 1) return 0;

		int surround = input[iedge.cols * (row - 1) + (col - 1)] +
			input[iedge.cols * (row - 1) + (col)] +
			input[iedge.cols * (row - 1) + (col + 1)] +
			input[iedge.cols * (row)+(col - 1)] +
			input[iedge.cols * (row)+(col + 1)] +
			input[iedge.cols * (row + 1) + (col - 1)] +
			input[iedge.cols * (row + 1) + (col)] +
			input[iedge.cols * (row - 1) + (col + 1)];

		if (surround == 8 * 255) return 255;
		if (surround == 0) return 255;

		return input[iedge.cols * (row)+(col)];


	}

	int contrastEdges(Mat &minput, Mat &mouput, int contrast) {

		unsigned char *input = (unsigned char*)(mouput.data);

		Mat mwork = minput.clone();

		unsigned char *workdata = (unsigned char*)(mwork.data);

		// Now find if other pixels inside are minimum
		for (int row = 2; row < minput.rows - 2; row++) {
			for (int col = 2; col < minput.cols - 2; col++) {
				if (isPixelMinimum(minput, row, col, contrast)) {
					workdata[mwork.cols * row + col] = 0;
				}
				else {
					workdata[mwork.cols * row + col] = 255;
				}
			}
		}
		// Set border of work matrix to white		
		for (int col = 0; col < minput.cols; col++) {
			for (int row = 0; row < 2; row++) {
				workdata[mwork.cols * row + col] = 255;
			}
			for (int row = minput.rows - 2; row < minput.rows; row++) {
				workdata[mwork.cols * row + col] = 255;
			}
		}
		for (int row = 0; row < minput.rows; row++) {
			for (int col = 0; col < 2; col++) {
				workdata[mwork.cols * row + col] = 255;
			}
			for (int col = minput.cols - 2; col < minput.cols; col++) {
				workdata[mwork.cols * row + col] = 255;
			}
		}
		// correct pixels

		for (int row = 2; row < mwork.rows - 2; row++) {
			for (int col = 2; col < mwork.cols - 2; col++) {
				input[mouput.cols * row + col] = correctPixel(mwork, row, col);
			}
		}
		// Set border of output matrix to white	

		for (int col = 0; col < mouput.cols; col++) {
			for (int row = 0; row < 2; row++) {
				input[mouput.cols * row + col] = 255;
			}
			for (int row = mouput.rows - 2; row < mouput.rows; row++) {
				input[mouput.cols * row + col] = 255;
			}
		}
		for (int row = 0; row < mouput.rows; row++) {
			for (int col = 0; col < 2; col++) {
				input[mouput.cols * row + col] = 255;
			}
			for (int col = mouput.cols - 2; col < mouput.cols; col++) {
				input[mouput.cols * row + col] = 255;
			}
		}
		return 0;
	}
	CV_EXPORTS_W  void BrightEdges(Mat &image, Mat &edge, int contrast, int shortrange, int longrange)
	{

		Mat gray, gblur, bblur, diff, cedge;

		GaussianBlur(image, gblur, Size(shortrange, shortrange), 0);

		blur(image, bblur, Size(longrange, longrange));

		absdiff(gblur, bblur, diff);

		cvtColor(diff, gray, COLOR_BGR2GRAY);

		equalizeHist(gray, cedge);

		edge = cedge.clone();

		if (contrast > 0) {
			contrastEdges(cedge, edge, contrast);
		}

	}
}
