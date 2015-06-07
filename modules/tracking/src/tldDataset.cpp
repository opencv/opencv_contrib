#include "opencv2/tracking/tldDataset.hpp"

namespace cv
{
	namespace tld
	{
		char tldRootPath[100];
		int frameNum = 0;
		bool flagPNG = false;

		cv::Rect2d tld_InitDataset(int datasetInd, char* rootPath)
		{
			char* folderName = "";
			int x, y, w, h;
			flagPNG = false;

			frameNum = 1;

			if (datasetInd == 1) {
				folderName = "01_david";
				x = 165, y = 83;
				w = 51; h = 54;
				frameNum = 100;
			}
			if (datasetInd == 2) {
				folderName = "02_jumping";
				x = 147, y = 110;
				w = 33; h = 32;
			}
			if (datasetInd == 3) {
				folderName = "03_pedestrian1";
				x = 47, y = 51;
				w = 21; h = 36;
			}
			if (datasetInd == 4) {
				folderName = "04_pedestrian2";
				x = 130, y = 134;
				w = 21; h = 53;
			}
			if (datasetInd == 5) {
				folderName = "05_pedestrian3";
				x = 154, y = 102;
				w = 24; h = 52;
			}
			if (datasetInd == 6) {
				folderName = "06_car";
				x = 142, y = 125;
				w = 90; h = 39;
			}
			if (datasetInd == 7) {
				folderName = "07_motocross";
				x = 290, y = 43;
				w = 23; h = 40;
				flagPNG = true;
			}
			if (datasetInd == 8) {
				folderName = "08_volkswagen";
				x = 273, y = 77;
				w = 27; h = 25;
			}
			if (datasetInd == 9) {
				folderName = "09_carchase";
				x = 145, y = 84;
				w = 54; h = 37;
			}
			if (datasetInd == 10){
				folderName = "10_panda";
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

			_itoa(frameNum, numStr, 10);
			strcat(fullPath, numStr);
			if (flagPNG) strcat(fullPath, ".png");
			else strcat(fullPath, ".jpg");
			frameNum++;

			return cv::imread(fullPath);
		}

	}
}