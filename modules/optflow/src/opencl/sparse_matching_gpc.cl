// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

// @Authors
//    Vladislav Samsonov, vvladxx@gmail.com

__kernel void getPatchDescriptor(
  __global const uchar* imgCh0, int ic0step, int ic0off,
  __global const uchar* imgCh1, int ic1step, int ic1off,
  __global const uchar* imgCh2, int ic2step, int ic2off,
  __global uchar* out, int outstep, int outoff,
  const int gh, const int gw, const int PR  )
{
  const int i = get_global_id(0);
  const int j = get_global_id(1);

  if (i >= gh || j >= gw)
    return;

  __global double* desc = (__global double*)(out + (outstep * (i * gw + j) + outoff));
  const int patchRadius = PR * 2;
  float patch[PATCH_RADIUS_DOUBLED][PATCH_RADIUS_DOUBLED];

  for (int i0 = 0; i0 < patchRadius; ++i0) {
    __global const float* ch0Row = (__global const float*)(imgCh0 + (ic0step * (i + i0) + ic0off + j * sizeof(float)));
    for (int j0 = 0; j0 < patchRadius; ++j0)
      patch[i0][j0] = ch0Row[j0];
  }

  #pragma unroll
  for (int n0 = 0; n0 < 4; ++n0) {
    #pragma unroll
    for (int n1 = 0; n1 < 4; ++n1) {
      double sum = 0;
      for (int i0 = 0; i0 < patchRadius; ++i0)
        for (int j0 = 0; j0 < patchRadius; ++j0)
          sum += patch[i0][j0] * cos(CV_PI * (i0 + 0.5) * n0 / patchRadius) * cos(CV_PI * (j0 + 0.5) * n1 / patchRadius);
      desc[n0 * 4 + n1] = sum / PR;
    }
  }

  for (int k = 0; k < 4; ++k) {
    desc[k] *= SQRT2_INV;
    desc[k * 4] *= SQRT2_INV;
  }

  double sum = 0;

  for (int i0 = 0; i0 < patchRadius; ++i0) {
    __global const float* ch1Row = (__global const float*)(imgCh1 + (ic1step * (i + i0) + ic1off + j * sizeof(float)));
    for (int j0 = 0; j0 < patchRadius; ++j0)
      sum += ch1Row[j0];
  }

  desc[16] = sum / patchRadius;
  sum = 0;

  for (int i0 = 0; i0 < patchRadius; ++i0) {
    __global const float* ch2Row = (__global const float*)(imgCh2 + (ic2step * (i + i0) + ic2off + j * sizeof(float)));
    for (int j0 = 0; j0 < patchRadius; ++j0)
      sum += ch2Row[j0];
  }

  desc[17] = sum / patchRadius;
}
