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

#include "precomp.hpp"
#include "opencv2/tracking/tldDataset.hpp"

namespace cv {
namespace detail {
inline namespace tracking {

	namespace tld
	{
		char tldRootPath[100];
		int frameNum = 0;
		bool flagPNG = false;
		bool flagVOT = false;

		//TLD Dataset Parameters
		const char* tldFolderName[10] = {
			"01_david",
			"02_jumping",
			"03_pedestrian1",
			"04_pedestrian2",
			"05_pedestrian3",
			"06_car",
			"07_motocross",
			"08_volkswagen",
			"09_carchase",
			"10_panda"
		};
		const  char* votFolderName[60] = {
			"bag", "ball1", "ball2", "basketball", "birds1", "birds2", "blanket", "bmx", "bolt1", "bolt2",
			"book", "butterfly", "car1", "car2", "crossing", "dinosaur", "fernando", "fish1", "fish2", "fish3",
			"fish4", "girl", "glove", "godfather", "graduate", "gymnastics1", "gymnastics2	", "gymnastics3", "gymnastics4", "hand",
			"handball1", "handball2", "helicopter", "iceskater1", "iceskater2", "leaves", "marching", "matrix", "motocross1", "motocross2",
			"nature", "octopus", "pedestrian1", "pedestrian2", "rabbit", "racing", "road", "shaking", "sheep", "singer1",
			"singer2", "singer3", "soccer1", "soccer2", "soldier", "sphere", "tiger", "traffic", "tunnel", "wiper"
		};

		const Rect2d tldInitBB[10] = {
			Rect2d(165, 93, 51, 54), Rect2d(147, 110, 33, 32), Rect2d(47, 51, 21, 36), Rect2d(130, 134, 21, 53), Rect2d(154, 102, 24, 52),
			Rect2d(142, 125, 90, 39), Rect2d(290, 43, 23, 40), Rect2d(273, 77, 27, 25), Rect2d(337, 219, 54, 37), Rect2d(58, 100, 27, 22)
		};
		const Rect2d votInitBB[60] = {
			Rect2d(142, 125, 90, 39), Rect2d(490, 400, 40, 40), Rect2d(273, 77, 27, 25), Rect2d(145, 84, 54, 37), Rect2d(58, 100, 27, 22),
			Rect2d(450, 380, 60, 60), Rect2d(290, 43, 23, 40), Rect2d(273, 77, 27, 25), Rect2d(225, 175, 50, 50), Rect2d(58, 100, 27, 22),

			Rect2d(142, 125, 90, 39), Rect2d(290, 43, 23, 40), Rect2d(273, 77, 27, 25), Rect2d(145, 84, 54, 37), Rect2d(560, 460, 50, 120),
			Rect2d(142, 125, 90, 39), Rect2d(290, 43, 23, 40), Rect2d(273, 77, 27, 25), Rect2d(145, 84, 54, 37), Rect2d(58, 100, 27, 22),

			Rect2d(142, 125, 90, 39), Rect2d(290, 43, 23, 40), Rect2d(273, 77, 27, 25), Rect2d(145, 84, 54, 37), Rect2d(58, 100, 27, 22),
			Rect2d(142, 125, 90, 39), Rect2d(290, 43, 23, 40), Rect2d(273, 77, 27, 25), Rect2d(145, 84, 54, 37), Rect2d(58, 100, 27, 22),

			Rect2d(142, 125, 90, 39), Rect2d(290, 43, 23, 40), Rect2d(273, 77, 27, 25), Rect2d(145, 84, 54, 37), Rect2d(58, 100, 27, 22),
			Rect2d(142, 125, 90, 39), Rect2d(290, 43, 23, 40), Rect2d(273, 77, 27, 25), Rect2d(145, 84, 54, 37), Rect2d(58, 100, 27, 22),

			Rect2d(142, 125, 90, 39), Rect2d(290, 43, 23, 40), Rect2d(273, 77, 27, 25), Rect2d(145, 84, 54, 37), Rect2d(58, 100, 27, 22),
			Rect2d(142, 125, 90, 39), Rect2d(290, 43, 23, 40), Rect2d(273, 77, 27, 25), Rect2d(145, 84, 54, 37), Rect2d(58, 100, 27, 22),

			Rect2d(142, 125, 90, 39), Rect2d(290, 43, 23, 40), Rect2d(273, 77, 27, 25), Rect2d(145, 84, 54, 37), Rect2d(58, 100, 27, 22),
			Rect2d(142, 125, 90, 39), Rect2d(290, 43, 23, 40), Rect2d(273, 77, 27, 25), Rect2d(145, 84, 54, 37), Rect2d(58, 100, 27, 22),
		};

		int tldFrameOffset[10] = { 100, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
		int votFrameOffset[60] = {
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1
		};
		bool tldFlagPNG[10] = { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 };
		bool votFlagPNG[60] = {
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		};

		cv::Rect2d tld_InitDataset(int videoInd, const char* rootPath, int datasetInd)
		{
			char* folderName = (char *)"";
			double x = 0,
				y = 0,
				w = 0,
				h = 0;

			//Index range
			// 1-10 TLD Dataset
			// 1-60 VOT 2015 Dataset
			int id = videoInd - 1;

			if (datasetInd == 0)
			{
				folderName = (char*)tldFolderName[id];
				x = tldInitBB[id].x;
				y = tldInitBB[id].y;
				w = tldInitBB[id].width;
				h = tldInitBB[id].height;
				frameNum = tldFrameOffset[id];
				flagPNG = tldFlagPNG[id];
				flagVOT = false;
			}
			if (datasetInd == 1)
				{
					folderName = (char*)votFolderName[id];
					x = votInitBB[id].x;
					y = votInitBB[id].y;
					w = votInitBB[id].width;
					h = votInitBB[id].height;
					frameNum = votFrameOffset[id];
					flagPNG = votFlagPNG[id];
					flagVOT = true;
				}

			strcpy(tldRootPath, rootPath);
			strcat(tldRootPath, "\\");
			strcat(tldRootPath, folderName);


			return cv::Rect2d(x, y, w, h);
		}

		cv::String tld_getNextDatasetFrame()
		{
			char fullPath[100];
			char numStr[10];
			strcpy(fullPath, tldRootPath);
			strcat(fullPath, "\\");
			if (flagVOT)
				strcat(fullPath, "000");
			if (frameNum < 10) strcat(fullPath, "0000");
			else if (frameNum < 100) strcat(fullPath, "000");
			else if (frameNum < 1000) strcat(fullPath, "00");
			else if (frameNum < 10000) strcat(fullPath, "0");

			sprintf(numStr, "%d", frameNum);
			strcat(fullPath, numStr);
			if (flagPNG) strcat(fullPath, ".png");
			else strcat(fullPath, ".jpg");
			frameNum++;

			return fullPath;
		}

	}

}}}
