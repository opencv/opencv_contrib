#pragma once

/*// gaussian 3x3 pattern, based on 'floor(fspecial('gaussian', 3, 1)*256)'
static const int s_nSamplesInitPatternWidth = 3;
static const int s_nSamplesInitPatternHeight = 3;
static const int s_nSamplesInitPatternTot = 256;
static const int s_anSamplesInitPattern[s_nSamplesInitPatternHeight][s_nSamplesInitPatternWidth] = {
	{19,    32,    19,},
	{32,    52,    32,},
	{19,    32,    19,},
};*/

// gaussian 7x7 pattern, based on 'floor(fspecial('gaussian',7,1)*4096)'
static const int s_nSamplesInitPatternWidth = 7;
static const int s_nSamplesInitPatternHeight = 7;
static const int s_nSamplesInitPatternTot = 4096;
static const int s_anSamplesInitPattern[s_nSamplesInitPatternHeight][s_nSamplesInitPatternWidth] = {
	{0,     0,     4,     7,     4,     0,     0,},
	{0,    11,    53,    88,    53,    11,     0,},
	{4,    53,   240,   399,   240,    53,     4,},
	{7,    88,   399,   660,   399,    88,     7,},
	{4,    53,   240,   399,   240,    53,     4,},
	{0,    11,    53,    88,    53,    11,     0,},
	{0,     0,     4,     7,     4,     0,     0,},
};

//! returns a random init/sampling position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
static inline void getRandSamplePosition(int& x_sample, int& y_sample, const int x_orig, const int y_orig, const int border, const cv::Size& imgsize) {
	int r = 1+rand()%s_nSamplesInitPatternTot;
	for(x_sample=0; x_sample<s_nSamplesInitPatternWidth; ++x_sample) {
		for(y_sample=0; y_sample<s_nSamplesInitPatternHeight; ++y_sample) {
			r -= s_anSamplesInitPattern[y_sample][x_sample];
			if(r<=0)
				goto stop;
		}
	}
	stop:
	x_sample += x_orig-s_nSamplesInitPatternWidth/2;
	y_sample += y_orig-s_nSamplesInitPatternHeight/2;
	if(x_sample<border)
		x_sample = border;
	else if(x_sample>=imgsize.width-border)
		x_sample = imgsize.width-border-1;
	if(y_sample<border)
		y_sample = border;
	else if(y_sample>=imgsize.height-border)
		y_sample = imgsize.height-border-1;
}

// simple 8-connected (3x3) neighbors pattern
static const int s_anNeighborPatternSize_3x3 = 8;
static const int s_anNeighborPattern_3x3[8][2] = {
	{-1, 1},  { 0, 1},  { 1, 1},
	{-1, 0},            { 1, 0},
	{-1,-1},  { 0,-1},  { 1,-1},
};

//! returns a random neighbor position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
static inline void getRandNeighborPosition_3x3(int& x_neighbor, int& y_neighbor, const int x_orig, const int y_orig, const int border, const cv::Size& imgsize) {
	int r = rand()%s_anNeighborPatternSize_3x3;
	x_neighbor = x_orig+s_anNeighborPattern_3x3[r][0];
	y_neighbor = y_orig+s_anNeighborPattern_3x3[r][1];
	if(x_neighbor<border)
		x_neighbor = border;
	else if(x_neighbor>=imgsize.width-border)
		x_neighbor = imgsize.width-border-1;
	if(y_neighbor<border)
		y_neighbor = border;
	else if(y_neighbor>=imgsize.height-border)
		y_neighbor = imgsize.height-border-1;
}

// 5x5 neighbors pattern
static const int s_anNeighborPatternSize_5x5 = 24;
static const int s_anNeighborPattern_5x5[24][2] = {
	{-2, 2},  {-1, 2},  { 0, 2},  { 1, 2},  { 2, 2},
	{-2, 1},  {-1, 1},  { 0, 1},  { 1, 1},  { 2, 1},
	{-2, 0},  {-1, 0},            { 1, 0},  { 2, 0},
	{-2,-1},  {-1,-1},  { 0,-1},  { 1,-1},  { 2,-1},
	{-2,-2},  {-1,-2},  { 0,-2},  { 1,-2},  { 2,-2},
};

//! returns a random neighbor position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
static inline void getRandNeighborPosition_5x5(int& x_neighbor, int& y_neighbor, const int x_orig, const int y_orig, const int border, const cv::Size& imgsize) {
	int r = rand()%s_anNeighborPatternSize_5x5;
	x_neighbor = x_orig+s_anNeighborPattern_5x5[r][0];
	y_neighbor = y_orig+s_anNeighborPattern_5x5[r][1];
	if(x_neighbor<border)
		x_neighbor = border;
	else if(x_neighbor>=imgsize.width-border)
		x_neighbor = imgsize.width-border-1;
	if(y_neighbor<border)
		y_neighbor = border;
	else if(y_neighbor>=imgsize.height-border)
		y_neighbor = imgsize.height-border-1;
}
