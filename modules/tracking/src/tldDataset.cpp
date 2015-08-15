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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
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

#include "opencv2/tracking/tldDataset.hpp"

namespace cv
{
	namespace tld
	{
		char tldRootPath[100];
		int frameNum = 0;
		bool flagPNG = false;

		cv::Rect2d tld_InitDataset(int datasetInd,const char* rootPath)
		{
			char* folderName = (char *)"";
			int x = 0;
			int y = 0;
			int w = 0;
			int h = 0;
			flagPNG = false;

			frameNum = 1;

			if (datasetInd == 1) {
				folderName = (char *)"01_david";
				x = 165, y = 83;
				w = 51; h = 54;
				frameNum = 100;
			}
			if (datasetInd == 2) {
				folderName = (char *)"02_jumping";
				x = 147, y = 110;
				w = 33; h = 32;
			}
			if (datasetInd == 3) {
				folderName = (char *)"03_pedestrian1";
				x = 47, y = 51;
				w = 21; h = 36;
			}
			if (datasetInd == 4) {
				folderName = (char *)"04_pedestrian2";
				x = 130, y = 134;
				w = 21; h = 53;
			}
			if (datasetInd == 5) {
				folderName = (char *)"05_pedestrian3";
				x = 154, y = 102;
				w = 24; h = 52;
			}
			if (datasetInd == 6) {
				folderName = (char *)"06_car";
				x = 142, y = 125;
				w = 90; h = 39;
			}
			if (datasetInd == 7) {
				folderName = (char *)"07_motocross";
				x = 290, y = 43;
				w = 23; h = 40;
				flagPNG = true;
			}
			if (datasetInd == 8) {
				folderName = (char *)"08_volkswagen";
				x = 273, y = 77;
				w = 27; h = 25;
			}
			if (datasetInd == 9) {
				folderName = (char *)"09_carchase";
				x = 145, y = 84;
				w = 54; h = 37;
			}
			if (datasetInd == 10){
				folderName = (char *)"10_panda";
				x = 58, y = 100;
				w = 27; h = 22;
			}

			strcpy(tldRootPath, rootPath);
			strcat(tldRootPath, "\\");
			strcat(tldRootPath, folderName);


			return cv::Rect2d(x, y, w, h);
		}

		cv::Mat tld_getNextDatasetFrame()
		{
			char fullPath[100];
			char numStr[10];
			strcpy(fullPath, tldRootPath);
			strcat(fullPath, "\\");
			if (frameNum < 10) strcat(fullPath, "0000");
			else if (frameNum < 100) strcat(fullPath, "000");
			else if (frameNum < 1000) strcat(fullPath, "00");
			else if (frameNum < 10000) strcat(fullPath, "0");

			sprintf(numStr, "%d", frameNum);
			strcat(fullPath, numStr);
			if (flagPNG) strcat(fullPath, ".png");
			else strcat(fullPath, ".jpg");
			frameNum++;

			return cv::imread(fullPath);
		}

	}
}