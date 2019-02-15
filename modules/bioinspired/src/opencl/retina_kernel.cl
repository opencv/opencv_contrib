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
// Copyright (C) 2010-2013, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
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
// This software is provided by the copyright holders and contributors as is and
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

//data (which is float) is aligend in 32 bytes
#define WIDTH_MULTIPLE (32 >> 2)

/////////////////////////////////////////////////////////
//------------------------------------------------------
// basicretinafilter
//////////////// _spatiotemporalLPfilter ////////////////
//_horizontalCausalFilter_addInput
kernel void horizontalCausalFilter_addInput(
    global const float * input,
    global float * output,
    const int cols,
    const int rows,
    const int elements_per_row,
    const int in_offset,
    const int out_offset,
    const float _tau,
    const float _a
)
{
    int gid = get_global_id(0);
    if(gid >= rows)
    {
        return;
    }

    global const float * iptr =
        input  + mad24(gid, elements_per_row, in_offset / 4);
    global float * optr =
        output + mad24(gid, elements_per_row, out_offset / 4);

    float res;
    float4 in_v4, out_v4, sum_v4, res_v4 = (float4)(0);
    //vectorize to increase throughput
    for(int i = 0; i < cols / 4; ++i, iptr += 4, optr += 4)
    {
        in_v4  = vload4(0, iptr);
        out_v4 = vload4(0, optr) * _tau;
        sum_v4 = in_v4 + out_v4;

        res_v4.x = sum_v4.x + _a * res_v4.w;
        res_v4.y = sum_v4.y + _a * res_v4.x;
        res_v4.z = sum_v4.z + _a * res_v4.y;
        res_v4.w = sum_v4.w + _a * res_v4.z;

        vstore4(res_v4, 0, optr);
    }

    optr = output + mad24(gid + 1, elements_per_row, -4 + out_offset / 4);
    res_v4 = (float4)(0);
    for(int i = 0; i < elements_per_row / 4; ++i, optr -= 4)
    {
        // shift left, `offset` is type `size_t` so it cannot be negative
        out_v4 = vload4(0, optr);

        res_v4.w = out_v4.w + _a * res_v4.x;
        res_v4.z = out_v4.z + _a * res_v4.w;
        res_v4.y = out_v4.y + _a * res_v4.z;
        res_v4.x = out_v4.x + _a * res_v4.y;

        vstore4(res_v4, 0, optr);
    }
}

//_verticalCausalFilter
kernel void verticalCausalFilter(
    global float * output,
    const int cols,
    const int rows,
    const int elements_per_row,
    const int out_offset,
    const float _a,
    const float _gain
)
{
    int gid = get_global_id(0) * 2;
    if(gid >= cols)
    {
        return;
    }

    global float * optr = output + gid + out_offset / 4;
    float2 input;
    float2 result = (float2)0;
    for(int i = 0; i < rows; ++i, optr += elements_per_row)
    {
        input = vload2(0, optr);
        result = input + _a * result;
        vstore2(result, 0, optr);
    }

    optr = output + (rows - 1) * elements_per_row + gid + out_offset / 4;
    result = (float2)0;
    for(int i = 0; i < rows; ++i, optr -= elements_per_row)
    {
        input = vload2(0, optr);
        result = input + _a * result;
        vstore2(_gain * result, 0, optr);
    }
}

kernel void verticalCausalFilter_multichannel(
    global float * output,
    const int cols,
    const int rows,
    const int elements_per_row,
    const int out_offset,
    const float _a,
    const float _gain
)
{
    int gid = get_global_id(0) * 2;
    if(gid >= cols)
    {
        return;
    }

    global float * optr[3];
    float2 input[3];
    float2 result[3] = { (float2)0, (float2)0, (float2)0 };

    optr[0] = output + gid + out_offset / 4;
    optr[1] = output + gid + out_offset / 4 + rows * elements_per_row;
    optr[2] = output + gid + out_offset / 4 + 2 * rows * elements_per_row;

    for(int i = 0; i < rows; ++i)
    {
        input[0] = vload2(0, optr[0]);
        input[1] = vload2(0, optr[1]);
        input[2] = vload2(0, optr[2]);

        result[0] = input[0] + _a * result[0];
        result[1] = input[1] + _a * result[1];
        result[2] = input[2] + _a * result[2];

        vstore2(result[0], 0, optr[0]);
        vstore2(result[1], 0, optr[1]);
        vstore2(result[2], 0, optr[2]);

        optr[0] += elements_per_row;
        optr[1] += elements_per_row;
        optr[2] += elements_per_row;
    }

    optr[0] = output + (rows - 1) * elements_per_row + gid + out_offset / 4;
    optr[1] = output + (rows - 1) * elements_per_row + gid + out_offset / 4 + rows * elements_per_row;
    optr[2] = output + (rows - 1) * elements_per_row + gid + out_offset / 4 + 2 * rows * elements_per_row;
    result[0] = result[1] = result[2] = (float2)0;

    for(int i = 0; i < rows; ++i)
    {
        input[0] = vload2(0, optr[0]);
        input[1] = vload2(0, optr[1]);
        input[2] = vload2(0, optr[2]);

        result[0] = input[0] + _a * result[0];
        result[1] = input[1] + _a * result[1];
        result[2] = input[2] + _a * result[2];

        vstore2(_gain * result[0], 0, optr[0]);
        vstore2(_gain * result[1], 0, optr[1]);
        vstore2(_gain * result[2], 0, optr[2]);

        optr[0] -= elements_per_row;
        optr[1] -= elements_per_row;
        optr[2] -= elements_per_row;
    }
}

//
// end of _spatiotemporalLPfilter
/////////////////////////////////////////////////////////////////////

//////////////// verticalCausalFilter_Irregular ////////////////
//////////////// verticalCausalFilter_Irregular ////////////////
kernel void verticalCausalFilter_Irregular(
    global float * output,
    global float * buffer,
    const int cols,
    const int rows,
    const int elements_per_row,
    const int out_offset,
    const int buffer_offset,
    const float gain
)
{
    int gid = get_global_id(0) * 2;
    if(gid >= cols)
    {
        return;
    }

    global float * optr[3];
    global float * bptr = buffer + gid + buffer_offset / 4;
    float2 result[3] = { (float2)0, (float2)0, (float2)0 };
    float2 grad, input[3];
    optr[0] = output + gid + out_offset / 4;
    optr[1] = output + gid + out_offset / 4 + rows * elements_per_row;
    optr[2] = output + gid + out_offset / 4 + 2 * rows * elements_per_row;
    for(int i = 0; i < rows; ++i, bptr += elements_per_row)
    {
        input[0] = vload2(0, optr[0]);
        input[1] = vload2(0, optr[1]);
        input[2] = vload2(0, optr[2]);
        grad = vload2(0, bptr);
        result[0] = input[0] + grad * result[0];
        result[1] = input[1] + grad * result[1];
        result[2] = input[2] + grad * result[2];
        vstore2(result[0], 0, optr[0]);
        vstore2(result[1], 0, optr[1]);
        vstore2(result[2], 0, optr[2]);
        optr[0] += elements_per_row;
        optr[1] += elements_per_row;
        optr[2] += elements_per_row;
    }

    int start_idx = mad24(rows - 1, elements_per_row, gid);
    optr[0] = output + start_idx + out_offset / 4;
    optr[1] = output + start_idx + out_offset / 4 + rows * elements_per_row;
    optr[2] = output + start_idx + out_offset / 4 + 2 * rows * elements_per_row;
    bptr = buffer + start_idx + buffer_offset / 4;
    result[0] = result[1] = result[2] = (float2)0;
    for(int i = 0; i < rows; ++i, bptr -= elements_per_row)
    {
        input[0] = vload2(0, optr[0]);
        input[1] = vload2(0, optr[1]);
        input[2] = vload2(0, optr[2]);
        grad = vload2(0, bptr);
        result[0] = input[0] + grad * result[0];
        result[1] = input[1] + grad * result[1];
        result[2] = input[2] + grad * result[2];
        vstore2(gain * result[0], 0, optr[0]);
        vstore2(gain * result[1], 0, optr[1]);
        vstore2(gain * result[2], 0, optr[2]);
        optr[0] -= elements_per_row;
        optr[1] -= elements_per_row;
        optr[2] -= elements_per_row;
    }
}

//////////////// _adaptiveHorizontalCausalFilter_addInput ////////////////
kernel void adaptiveHorizontalCausalFilter_addInput(
    global const float * input,
    global const float * gradient,
    global float * output,
    const int cols,
    const int rows,
    const int elements_per_row,
    const int in_offset,
    const int grad_offset,
    const int out_offset
)
{
    int gid = get_global_id(0);
    if(gid >= rows)
    {
        return;
    }

    global const float * iptr =
        input + mad24(gid, elements_per_row, in_offset / 4);
    global const float * gptr =
        gradient + mad24(gid, elements_per_row, grad_offset / 4);
    global float * optr =
        output + mad24(gid, elements_per_row, out_offset / 4);

    float4 in_v4, grad_v4, out_v4, res_v4 = (float4)(0);
    for(int i = 0; i < cols / 4; ++i, iptr += 4, gptr += 4, optr += 4)
    {
        in_v4   = vload4(0, iptr);
        grad_v4 = vload4(0, gptr);

        res_v4.x = in_v4.x + grad_v4.x * res_v4.w;
        res_v4.y = in_v4.y + grad_v4.y * res_v4.x;
        res_v4.z = in_v4.z + grad_v4.z * res_v4.y;
        res_v4.w = in_v4.w + grad_v4.w * res_v4.z;

        vstore4(res_v4, 0, optr);
    }

    optr = output + mad24(gid + 1, elements_per_row, -4 + out_offset / 4);
    gptr = gradient + mad24(gid + 1, elements_per_row, -4 + grad_offset / 4);
    res_v4 = (float4)(0);

    for(int i = 0; i < cols / 4; ++i, gptr -= 4, optr -= 4)
    {
        grad_v4 = vload4(0, gptr);
        out_v4 = vload4(0, optr);

        res_v4.w = out_v4.w + grad_v4.w * res_v4.x;
        res_v4.z = out_v4.z + grad_v4.z * res_v4.w;
        res_v4.y = out_v4.y + grad_v4.y * res_v4.z;
        res_v4.x = out_v4.x + grad_v4.x * res_v4.y;

        vstore4(res_v4, 0, optr);
    }
}

//////////////// _localLuminanceAdaptation ////////////////
// FIXME:
//  This kernel seems to have precision problem on GPU
kernel void localLuminanceAdaptation(
    global const float * luma,
    global const float * input,
    global float * output,
    const int cols,
    const int rows,
    const int elements_per_row,
    const float _localLuminanceAddon,
    const float _localLuminanceFactor,
    const float _maxInputValue
)
{
    int gidx = get_global_id(0) * 4, gidy = get_global_id(1);
    if(gidx >= cols || gidy >= rows)
    {
        return;
    }
    int offset = mad24(gidy, elements_per_row, gidx);
    float4 luma_vec = vload4(0, luma + offset);
    float4 X0 = luma_vec * _localLuminanceFactor + _localLuminanceAddon;
    float4 input_val = vload4(0, input + offset);
    // output of the following line may be different between GPU and CPU
    float4 out_vec = (_maxInputValue + X0) * input_val / (input_val + X0 + 0.00000000001f);
    vstore4(out_vec, 0, output + offset);
}
// end of basicretinafilter
//------------------------------------------------------
/////////////////////////////////////////////////////////



/////////////////////////////////////////////////////////
//------------------------------------------------------
// magno
// TODO: this kernel has too many buffer accesses, better to make it
//   vector read/write for fetch efficiency
kernel void amacrineCellsComputing(
    global const float * opl_on,
    global const float * opl_off,
    global float * prev_in_on,
    global float * prev_in_off,
    global float * out_on,
    global float * out_off,
    const int cols,
    const int rows,
    const int elements_per_row,
    const float coeff
)
{
    int gidx = get_global_id(0) * 4, gidy = get_global_id(1);
    if(gidx >= cols || gidy >= rows)
    {
        return;
    }

    int offset = mad24(gidy, elements_per_row, gidx);
    opl_on      += offset;
    opl_off     += offset;
    prev_in_on  += offset;
    prev_in_off += offset;
    out_on      += offset;
    out_off     += offset;

    float4 val_opl_on = vload4(0, opl_on);
    float4 val_opl_off = vload4(0, opl_off);

    float4 magnoXonPixelResult = coeff * (vload4(0, out_on) + val_opl_on - vload4(0, prev_in_on));
    vstore4(fmax(magnoXonPixelResult, 0), 0, out_on);
    float4 magnoXoffPixelResult = coeff * (vload4(0, out_off) + val_opl_off - vload4(0, prev_in_off));
    vstore4(fmax(magnoXoffPixelResult, 0), 0, out_off);

    vstore4(val_opl_on, 0, prev_in_on);
    vstore4(val_opl_off, 0, prev_in_off);
}

/////////////////////////////////////////////////////////
//------------------------------------------------------
// parvo
// TODO: this kernel has too many buffer accesses, needs optimization
kernel void OPL_OnOffWaysComputing(
    global float4 * photo_out,
    global float4 * horiz_out,
    global float4 * bipol_on,
    global float4 * bipol_off,
    global float4 * parvo_on,
    global float4 * parvo_off,
    const int cols,
    const int rows,
    const int elements_per_row
)
{
    int gidx = get_global_id(0), gidy = get_global_id(1);
    if(gidx * 4 >= cols || gidy >= rows)
    {
        return;
    }
    // we assume elements_per_row must be multiples of 4
    int offset = mad24(gidy, elements_per_row >> 2, gidx);
    photo_out += offset;
    horiz_out += offset;
    bipol_on  += offset;
    bipol_off += offset;
    parvo_on  += offset;
    parvo_off += offset;

    float4 diff = *photo_out - *horiz_out;
    float4 isPositive = convert_float4(abs(diff > (float4)0.0f));
    float4 res_on  = isPositive * diff;
    float4 res_off = (isPositive - (float4)(1.0f)) * diff;

    *bipol_on = res_on;
    *parvo_on = res_on;

    *bipol_off = res_off;
    *parvo_off = res_off;
}

/////////////////////////////////////////////////////////
//------------------------------------------------------
// retinacolor
inline int bayerSampleOffset(int step, int rows, int x, int y)
{
    return mad24(y, step, x) +
           ((y % 2) + (x % 2)) * rows * step;
}


/////// colorMultiplexing //////
kernel void runColorMultiplexingBayer(
    global const float * input,
    global float * output,
    const int cols,
    const int rows,
    const int elements_per_row
)
{
    int gidx = get_global_id(0) * 4, gidy = get_global_id(1);
    if(gidx >= cols || gidy >= rows)
    {
        return;
    }

    int offset = mad24(gidy, elements_per_row, gidx);
    float4 val;
    val.x = input[bayerSampleOffset(elements_per_row, rows, gidx + 0, gidy)];
    val.y = input[bayerSampleOffset(elements_per_row, rows, gidx + 1, gidy)];
    val.z = input[bayerSampleOffset(elements_per_row, rows, gidx + 2, gidy)];
    val.w = input[bayerSampleOffset(elements_per_row, rows, gidx + 3, gidy)];
    vstore4(val, 0, output + offset);
}

kernel void runColorDemultiplexingBayer(
    global const float * input,
    global float * output,
    const int cols,
    const int rows,
    const int elements_per_row
)
{
    int gidx = get_global_id(0) * 4, gidy = get_global_id(1);
    if(gidx >= cols || gidy >= rows)
    {
        return;
    }

    int offset = mad24(gidy, elements_per_row, gidx);
    float4 val = vload4(0, input + offset);
    output[bayerSampleOffset(elements_per_row, rows, gidx + 0, gidy)] = val.x;
    output[bayerSampleOffset(elements_per_row, rows, gidx + 1, gidy)] = val.y;
    output[bayerSampleOffset(elements_per_row, rows, gidx + 2, gidy)] = val.z;
    output[bayerSampleOffset(elements_per_row, rows, gidx + 3, gidy)] = val.w;
}

kernel void demultiplexAssign(
    global const float * input,
    global float * output,
    const int cols,
    const int rows,
    const int elements_per_row
)
{
    int gidx = get_global_id(0), gidy = get_global_id(1);
    if(gidx >= cols || gidy >= rows)
    {
        return;
    }

    int offset = bayerSampleOffset(elements_per_row, rows, gidx, gidy);
    output[offset] = input[offset];
}


//// normalizeGrayOutputCentredSigmoide
kernel void normalizeGrayOutputCentredSigmoide(
    global const float * input,
    global float * output,
    const int cols,
    const int rows,
    const int elements_per_row,
    const float meanval,
    const float X0
)

{
    int gidx = get_global_id(0) * 4, gidy = get_global_id(1);
    if(gidx >= cols || gidy >= rows)
    {
        return;
    }
    int offset = mad24(gidy, elements_per_row, gidx);

    float4 input_val = vload4(0, input + offset);
    input_val =  meanval + (meanval + X0) * (input_val - meanval) / (fabs(input_val - meanval) + X0);
    vstore4(input_val, 0, output + offset);
}

//// normalize by photoreceptors density
kernel void normalizePhotoDensity(
    global const float * chroma,
    global const float * colorDensity,
    global const float * multiplex,
    global float * luma,
    global float * demultiplex,
    const int cols,
    const int rows,
    const int elements_per_row,
    const float pG
)
{
    const int gidx = get_global_id(0) * 4, gidy = get_global_id(1);
    if(gidx >= cols || gidy >= rows)
    {
        return;
    }
    const int offset = mad24(gidy, elements_per_row, gidx);
    int index = offset;

    float4 Cr = vload4(0, chroma + index) * vload4(0, colorDensity + index);
    index += elements_per_row * rows;
    float4 Cg = vload4(0, chroma + index) * vload4(0, colorDensity + index);
    index += elements_per_row * rows;
    float4 Cb = vload4(0, chroma + index) * vload4(0, colorDensity + index);

    const float4 luma_res = (Cr + Cg + Cb) * pG;
    vstore4(luma_res, 0, luma + offset);
    float4 res_v4 = vload4(0, multiplex + offset) - luma_res;
    demultiplex[bayerSampleOffset(elements_per_row, rows, gidx + 0, gidy)] = res_v4.x;
    demultiplex[bayerSampleOffset(elements_per_row, rows, gidx + 1, gidy)] = res_v4.y;
    demultiplex[bayerSampleOffset(elements_per_row, rows, gidx + 2, gidy)] = res_v4.z;
    demultiplex[bayerSampleOffset(elements_per_row, rows, gidx + 3, gidy)] = res_v4.w;
}



//////// computeGradient ///////
// TODO:
// this function maybe accelerated by image2d_t or lds
kernel void computeGradient(
    global const float * luma,
    global float * gradient,
    const int cols,
    const int rows,
    const int elements_per_row
)
{
    int gidx = get_global_id(0) + 2, gidy = get_global_id(1) + 2;
    if(gidx >= cols - 2 || gidy >= rows - 2)
    {
        return;
    }
    int offset = mad24(gidy, elements_per_row, gidx);
    luma += offset;

    // horizontal and vertical local gradients
    const float v_grad = fabs(luma[elements_per_row] - luma[- elements_per_row]);
    const float h_grad = fabs(luma[1] - luma[-1]);

    // neighborhood horizontal and vertical gradients
    const float cur_val  = luma[0];
    const float v_grad_p = fabs(cur_val - luma[- 2 * elements_per_row]);
    const float h_grad_p = fabs(cur_val - luma[- 2]);
    const float v_grad_n = fabs(cur_val - luma[2 * elements_per_row]);
    const float h_grad_n = fabs(cur_val - luma[2]);

    const float horiz_grad = 0.5f * h_grad + 0.25f * (h_grad_p + h_grad_n);
    const float verti_grad = 0.5f * v_grad + 0.25f * (v_grad_p + v_grad_n);
    const bool is_vertical_greater = (horiz_grad < verti_grad) &&
                                     ((verti_grad - horiz_grad) > 1e-5);

    gradient[offset + elements_per_row * rows] = is_vertical_greater ? 0.06f : 0.57f;
    gradient[offset                          ] = is_vertical_greater ? 0.57f : 0.06f;
}


/////// substractResidual ///////
kernel void substractResidual(
    global float * input,
    const int cols,
    const int rows,
    const int elements_per_row,
    const float pR,
    const float pG,
    const float pB
)
{
    const int gidx = get_global_id(0) * 4, gidy = get_global_id(1);
    if(gidx >= cols || gidy >= rows)
    {
        return;
    }
    int indices [3] =
    {
        mad24(gidy, elements_per_row, gidx),
        mad24(gidy + rows, elements_per_row, gidx),
        mad24(gidy + 2 * rows, elements_per_row, gidx)
    };
    float4 vals[3];
    vals[0] = vload4(0, input + indices[0]);
    vals[1] = vload4(0, input + indices[1]);
    vals[2] = vload4(0, input + indices[2]);

    float4 residu = pR * vals[0] + pG * vals[1] + pB * vals[2];
    vstore4(vals[0] - residu, 0, input + indices[0]);
    vstore4(vals[1] - residu, 0, input + indices[1]);
    vstore4(vals[2] - residu, 0, input + indices[2]);
}

///// clipRGBOutput_0_maxInputValue /////
kernel void clipRGBOutput_0_maxInputValue(
    global float * input,
    const int cols,
    const int rows,
    const int elements_per_row,
    const float maxVal
)
{
    const int gidx = get_global_id(0) * 4, gidy = get_global_id(1);
    if(gidx >= cols || gidy >= rows)
    {
        return;
    }
    const int offset = mad24(gidy, elements_per_row, gidx);
    float4 val = vload4(0, input + offset);
    val = clamp(val, 0.0f, maxVal);
    vstore4(val, 0, input + offset);
}

//// normalizeGrayOutputNearZeroCentreredSigmoide ////
kernel void normalizeGrayOutputNearZeroCentreredSigmoide(
    global float * input,
    global float * output,
    const int cols,
    const int rows,
    const int elements_per_row,
    const float maxVal,
    const float X0cube
)
{
    const int gidx = get_global_id(0) * 4, gidy = get_global_id(1);
    if(gidx >= cols || gidy >= rows)
    {
        return;
    }
    const int offset = mad24(gidy, elements_per_row, gidx);
    float4 currentCubeLuminance = vload4(0, input + offset);
    currentCubeLuminance = currentCubeLuminance * currentCubeLuminance * currentCubeLuminance;
    float4 val = currentCubeLuminance * X0cube / (X0cube + currentCubeLuminance);
    vstore4(val, 0, output + offset);
}

//// centerReductImageLuminance ////
kernel void centerReductImageLuminance(
    global float * input,
    const int cols,
    const int rows,
    const int elements_per_row,
    const float mean,
    const float std_dev
)
{
    const int gidx = get_global_id(0) * 4, gidy = get_global_id(1);
    if(gidx >= cols || gidy >= rows)
    {
        return;
    }
    const int offset = mad24(gidy, elements_per_row, gidx);

    float4 val = vload4(0, input + offset);
    val = (val - mean) / std_dev;
    vstore4(val, 0, input + offset);
}

//// inverseValue ////
kernel void inverseValue(
    global float * input,
    const int cols,
    const int rows,
    const int elements_per_row
)
{
    const int gidx = get_global_id(0) * 4, gidy = get_global_id(1);
    if(gidx >= cols || gidy >= rows)
    {
        return;
    }
    const int offset = mad24(gidy, elements_per_row, gidx);
    float4 val = vload4(0, input + offset);
    val = 1.f / val;
    vstore4(val, 0, input + offset);
}

#define CV_PI 3.1415926535897932384626433832795

//// _processRetinaParvoMagnoMapping ////
kernel void processRetinaParvoMagnoMapping(
    global float * parvo,
    global float * magno,
    global float * output,
    const int cols,
    const int rows,
    const int halfCols,
    const int halfRows,
    const int elements_per_row,
    const float minDistance
)
{
    const int gidx = get_global_id(0), gidy = get_global_id(1);
    if(gidx >= cols || gidy >= rows)
    {
        return;
    }
    const int offset = mad24(gidy, elements_per_row, gidx);

    float distanceToCenter =
        sqrt(((float)(gidy - halfRows) * (gidy - halfRows) + (gidx - halfCols) * (gidx - halfCols)));

    float a = distanceToCenter < minDistance ?
              (0.5f + 0.5f * (float)cos(CV_PI * distanceToCenter / minDistance)) : 0;
    float b = 1.f - a;

    output[offset] = parvo[offset] * a + magno[offset] * b;
}
