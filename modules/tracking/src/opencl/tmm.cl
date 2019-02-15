// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#define LOCAL_SIZE_X 64
#define BLOCK_SIZE_X  3

__kernel void tmm(__global float *A, int m, int n, float alpha, __global float *D)
{
  int lidX = get_local_id(0);
  uint lsizeX = get_local_size(0);

  uint matI = get_group_id(1);
  uint matJ = get_group_id(0);

  if (matI < matJ)
    return;

  __local float4 a[LOCAL_SIZE_X], b[LOCAL_SIZE_X];
  float4 result;
  __local uint cnt;
  result = 0;
  cnt = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  do {
    // load block data to SLM.
    int global_block_base = (lidX + cnt * lsizeX) * BLOCK_SIZE_X;
    float4 pa[BLOCK_SIZE_X], pb[BLOCK_SIZE_X];

    #pragma unroll
    for(uint j = 0; j < BLOCK_SIZE_X && (cnt * lsizeX + lidX) * BLOCK_SIZE_X < n / 4; j++) {
      pa[j] = *(__global float4*)&A[matI * n + (global_block_base + j) * 4];
      if (matI != matJ)
        pb[j] = *(__global float4*)&A[matJ * n + (global_block_base + j) * 4];
      else
        pb[j] = pa[j];
    }

    // zero the data out-of-boundary.
    if (global_block_base + BLOCK_SIZE_X - 1 >= n/4) {
      #pragma unroll
      for(int i = 0; i < BLOCK_SIZE_X; i++) {
        if (global_block_base + i >= n/4)
          pb[i] = 0;
      }
    }

    pb[0] *= pa[0];

    for(int j = 1; j < BLOCK_SIZE_X; j++)
      pb[0] =  fma(pb[j], pa[j], pb[0]);

    b[lidX] = pb[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // perform reduce add
    for(int offset = LOCAL_SIZE_X / 2; offset > 0; offset >>= 1) {
      if (lidX < offset)
          b[lidX] += b[(lidX + offset)];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lidX == 0) {
      result += b[0];
      cnt++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  } while(cnt * BLOCK_SIZE_X * lsizeX < n / 4);
  if (lidX == 0) {
    float ret = (result.s0 + result.s1 + result.s2 + result.s3) * alpha;
    D[matI * m + matJ] = ret;
    if (matI != matJ)
      D[matJ * m + matI] = ret;
  }
}
