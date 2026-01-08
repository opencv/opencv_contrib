// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.



__kernel void NCC(__global const uchar *patch,
				 __global const uchar *positiveSamples,
				 __global const uchar *negativeSamples,
				 __global float *ncc,
				 int posNum,
				 int negNum)
{
	int id = get_global_id(0);
	if (id >= 1000) return;
	bool posFlg;

	if (id < 500)
		posFlg = true;
	if (id >= 500)
	{
		//Negative index
		id = id - 500;
		posFlg = false;
	}

	//Variables
	int s1 = 0, s2 = 0, n1 = 0, n2 = 0, prod = 0;
	float sq1 = 0, sq2 = 0, ares = 0;
	int N = 225;
	//NCC with positive sample
	if (posFlg && id < posNum)
	{
		for (int i = 0; i < N; i++)
		{
			s1 += positiveSamples[id * N + i];
			s2 += patch[i];
			n1 += positiveSamples[id * N + i] * positiveSamples[id * N + i];
			n2 += patch[i] * patch[i];
			prod += positiveSamples[id * N + i] * patch[i];
		}
		sq1 = sqrt(max(0.0, n1 - 1.0 * s1 * s1 / N));
		sq2 = sqrt(max(0.0, n2 - 1.0 * s2 * s2 / N));
		ares = (sq2 == 0) ? sq1 / fabs(sq1) : (prod - s1 * s2 / N) / sq1 / sq2;
		ncc[id] = ares;
	}

	//NCC with negative sample
	if (!posFlg && id < negNum)
	{
		for (int i = 0; i < N; i++)
		{

			s1 += negativeSamples[id * N + i];
			s2 += patch[i];
			n1 += negativeSamples[id * N + i] * negativeSamples[id * N + i];
			n2 += patch[i] * patch[i];
			prod += negativeSamples[id * N + i] * patch[i];
		}
		sq1 = sqrt(max(0.0, n1 - 1.0 * s1 * s1 / N));
		sq2 = sqrt(max(0.0, n2 - 1.0 * s2 * s2 / N));
		ares = (sq2 == 0) ? sq1 / fabs(sq1) : (prod - s1 * s2 / N) / sq1 / sq2;
		ncc[id+500] = ares;
	}
}

__kernel void batchNCC(__global const uchar *patches,
	__global const uchar *positiveSamples,
	__global const uchar *negativeSamples,
	__global float *posNcc,
	__global float *negNcc,
	int posNum,
	int negNum,
	int patchNum)
{
	int id = get_global_id(0);
	bool posFlg;

	if (id < 500*patchNum)
		posFlg = true;
	if (id >= 500*patchNum)
	{
		//Negative index
		id = id - 500*patchNum;
		posFlg = false;
	}

	int modelSampleID = id % 500;
	int patchID = id / 500;

	//Variables
	int s1 = 0, s2 = 0, n1 = 0, n2 = 0, prod = 0;
	float sq1 = 0, sq2 = 0, ares = 0;
	int N = 225;

	//NCC with positive sample
	if (posFlg && modelSampleID < posNum)
	{
		for (int i = 0; i < N; i++)
		{
			s1 += positiveSamples[modelSampleID * N + i];
			s2 += patches[patchID*N + i];
			n1 += positiveSamples[modelSampleID * N + i] * positiveSamples[modelSampleID * N + i];
			n2 += patches[patchID*N + i] * patches[patchID*N + i];
			prod += positiveSamples[modelSampleID * N + i] * patches[patchID*N + i];
		}
		sq1 = sqrt(max(0.0, n1 - 1.0 * s1 * s1 / N));
		sq2 = sqrt(max(0.0, n2 - 1.0 * s2 * s2 / N));
		ares = (sq2 == 0) ? sq1 / fabs(sq1) : (prod - s1 * s2 / N) / sq1 / sq2;
		posNcc[id] = ares;
	}

	//NCC with negative sample
	if (!posFlg && modelSampleID < negNum)
	{
		for (int i = 0; i < N; i++)
		{

			s1 += negativeSamples[modelSampleID * N + i];
			s2 += patches[patchID*N + i];
			n1 += negativeSamples[modelSampleID * N + i] * negativeSamples[modelSampleID * N + i];
			n2 += patches[patchID*N + i] * patches[patchID*N + i];
			prod += negativeSamples[modelSampleID * N + i] * patches[patchID*N + i];
		}
		sq1 = sqrt(max(0.0, n1 - 1.0 * s1 * s1 / N));
		sq2 = sqrt(max(0.0, n2 - 1.0 * s2 * s2 / N));
		ares = (sq2 == 0) ? sq1 / fabs(sq1) : (prod - s1 * s2 / N) / sq1 / sq2;
		negNcc[id] = ares;
	}
}
