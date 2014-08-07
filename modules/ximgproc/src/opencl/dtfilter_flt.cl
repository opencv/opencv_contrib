#if defined(_MSC_VER) //for correct syntax highlighting
#include "OpenCLKernel.hpp"
#define cn 3
#define SrcVec float3
#define NUM_ITERS 3
#endif

#define getFloatn(n, addr, index)           vload##n(index, (__global const float*)(addr))
#define storeFloatn(n, val, addr, index)    vstore##n(val, index, (__global float*)(addr))

#define PtrAdd(addr, bytesNum, Type)        ((__global Type *)((__global uchar*)(addr) + bytesNum))
#define PtrAddConst(addr, bytesNum, Type)   ((__global const Type *)((__global const uchar*)(addr) + bytesNum))

#if cn==3
#define ELEM_SIZE (int)(3*sizeof(float))
#define getPix(addr, index)         vload3(index, (__global const float*)(addr))
#define storePix(val, addr, index)  vstore3(val, index, (__global float*)(addr))
#else
#define ELEM_SIZE (int)sizeof(SrcVec)
#define getPix(addr, index)         ( *(__global const SrcVec*)(addr + (index)*ELEM_SIZE) ) 
#define storePix(val, addr, index)  ( *(__global SrcVec*)(addr + (index)*ELEM_SIZE) = val )
#endif

#define getFloat(addr, index)           ( *(__global const float*)(addr + (index)*sizeof(float)) )
#define storeFloat(val, addr, index)    ( *(__global float*)(addr + (index)*sizeof(float)) = val )

#define getInt(addr, index)             ( *(__global const int*)(addr + (index)*sizeof(int)) )
#define storeInt(val, addr, index)      ( *(__global int*)(addr + (index)*sizeof(int)) = val )

#define getFloat4(addr, index)          getFloatn(4, addr, index)
#define storeFloat4(val, addr, index)   storeFloatn(4, val, addr, index)

#define NC_USE_INTEGRAL_SRC
#undef NC_USE_INTEGRAL_SRC

__kernel void integrate_cols_4f(__global uchar *src, int src_step, int src_offset,
                                __global uchar *isrc, int isrc_step, int isrc_offset,
                                int rows, int col_chunks)
{
    int j = get_global_id(0);

    if ( !(j >= 0 && j < col_chunks) )
        return;

    src  += mad24(j, (int)sizeof(float4), src_offset);
    isrc += mad24(j, (int)sizeof(float4), isrc_offset);

    float4 sum = 0;
    storeFloat4(0, isrc, 0);
    isrc += isrc_step;

    for (int i = 0; i < rows; i++, src += src_step, isrc += isrc_step)
    {
        sum += getFloat4(src, 0);
        storeFloat4(sum, isrc, 0);
    }
}

__kernel void integrate_cols_with_dist(__global const uchar *src, int src_step, int src_offset,
                                       int src_rows, int src_cols,
                                       __global const uchar *dist, int dist_step, int dist_offset,
                                       __global       uchar *isrc, int isrc_step, int isrc_offset)
{
    int j = get_global_id(0);

    if ( !(j >= 0 && j < src_cols) )
        return;

    src  += mad24(j, ELEM_SIZE, src_offset);
    isrc += mad24(j, ELEM_SIZE, isrc_offset);
    dist += mad24(j, (int)sizeof(float), dist_offset);

    SrcVec sum = 0;
    storePix(0, isrc, 0);
    isrc += isrc_step;

    for (int i = 0; i < src_rows; i++, src += src_step, isrc += isrc_step, dist += dist_step)
    {
        sum += 0.5f * (getPix(src, 0) + getPix(src + src_step, 0)) * getFloat(dist, 0);
        storePix(sum, isrc, 0);
    }
}

__kernel void filter_IC_hor(__global const uchar *src, int src_step, int src_offset, int rows, int cols,
                                    __global const uchar *isrc, int isrc_step, int isrc_offset,
                                    __global const int   *bounds, int bounds_step, int bounds_offset,
                                    __global const float *tail, int tail_step, int tail_offset,
                                    __global       float *dist, int dist_step, int dist_offset,
                                    __global       uchar *dst, int dst_step, int dst_offset,
                                    float radius, int iterNum
                                   )
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (!(i >= 0 && i < rows) || !(j >= 0 && j < cols))
        return;

    dst  += mad24(i, dst_step, dst_offset);
    isrc += mad24(i, isrc_step, isrc_offset);
    src  += mad24(i, src_step, src_offset);
    
    int ioffset = mad24(iterNum*rows + i, bounds_step, mad24(j, 2*(int)sizeof(int), bounds_offset));
    bounds = PtrAddConst(bounds, ioffset, int);
    int il = bounds[0];
    int ir = bounds[1];

    int toffset = mad24(iterNum*rows + i, tail_step, mad24(j, 2*(int)sizeof(float), tail_offset));
    tail = PtrAddConst(tail, toffset, float);
    float tl = tail[0];
    float tr = tail[1];

    dist = PtrAdd(dist, mad24(i, dist_step, dist_offset), float);
    float dl = tl / dist[il-1];
    float dr = tr / dist[ir];

    SrcVec res;
    res  = getPix(isrc, ir) - getPix(isrc, il);                                 //center part
    res += 0.5f*tl * ( dl*getPix(src, il - 1) + (2.0f - dl)*getPix(src, il) );  //left tail
    res += 0.5f*tr * ( dr*getPix(src, ir + 1) + (2.0f - dr)*getPix(src, ir) );  //right tail
    res /= radius;

    storePix(res, dst, j);
}

__kernel void filter_NC_hor_by_bounds(__global const uchar *isrc, int isrc_step, int isrc_offset,
                                      __global const uchar *bounds, int bounds_step, int bounds_offset,
                                      __global       uchar *dst, int dst_step, int dst_offset,
                                      int rows, int cols,
                                      int iterNum)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (!(i >= 0 && i < rows) || !(j >= 0 && j < cols))
        return;

    dst  += mad24(i, dst_step, dst_offset);
    isrc += mad24(i, isrc_step, isrc_offset);
    
    int ioffset = mad24(iterNum*rows + i, bounds_step, mad24(j, 2*(int)sizeof(int), bounds_offset));
    __global const int *index = PtrAddConst(bounds, ioffset, int);
    int li = index[0];
    int hi = index[1];

    SrcVec res = (getPix(isrc, hi+1) - getPix(isrc, li)) / (float)(hi - li + 1);
    storePix(res, dst, j);
}

__kernel void filter_NC_hor(__global       uchar *src, int src_step, int src_offset, int src_rows, int src_cols,
                            __global const uchar *idist, int idist_step, int idist_offset,
                            __global       uchar *dst, int dst_step, int dst_offset,
                            float radius)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (!(i >= 0 && i < src_rows) || !(j >= 0 && j < src_cols))
        return;

    src   += mad24(i, src_step, src_offset);
    idist += mad24(i, idist_step, idist_offset);
    dst   += mad24(i, dst_step, dst_offset);

    int low_bound = j;
    int high_bound = j;
    float cur_idt_val = getFloat(idist, j);
    float low_idt_val = cur_idt_val - radius;
    float high_idt_val = cur_idt_val + radius;
    SrcVec sum = getPix(src, j);

    while (low_bound > 0 && low_idt_val < getFloat(idist, low_bound-1))
    {
        low_bound--;
        sum += getPix(src, low_bound);
    }

    while (getFloat(idist, high_bound + 1) < high_idt_val)
    {
        high_bound++;
        sum += getPix(src, high_bound);
    }

    storePix(sum / (float)(high_bound - low_bound + 1), dst, j);
}

__kernel void filter_NC_vert(__global const uchar *src, int src_step, int src_offset, int src_rows, int src_cols,
                             __global const uchar *idist, int idist_step, int idist_offset,
                             __global       uchar *dst, int dst_step, int dst_offset,
                             float radius)
{
    int j = get_global_id(0);

    if (!(j >= 0 && j < src_cols))
        return;

    __global const uchar *src_col   = src   + mad24(j, (int)ELEM_SIZE, src_offset);
    __global       uchar *dst_col   = dst   + mad24(j, (int)ELEM_SIZE, dst_offset);
    __global const uchar *idist_col = idist + mad24(j, (int)sizeof(float), idist_offset);

    int low_bound = 0, high_bound = 0;
    SrcVec sum = getPix(src_col, 0);
    SrcVec res;

    for (int i = 0; i < src_rows; i++)
    {
        float cur_idt_value = getFloat(idist_col + i*idist_step, 0);
        float low_idt_value = cur_idt_value - radius;
        float high_idt_value = cur_idt_value + radius; 

        while (getFloat(idist_col + low_bound*idist_step, 0) < low_idt_value)
        {
            sum -= getPix(src_col + low_bound*src_step, 0);
            low_bound++;
        }

        while (getFloat(idist_col + (high_bound + 1)*idist_step, 0) < high_idt_value)
        {
            high_bound++;
            sum += getPix(src_col + high_bound*src_step, 0);
        }

        res = sum / (float)(high_bound - low_bound + 1);
        storePix(res, dst_col + i*dst_step, 0);
    }
}

__kernel void filter_RF_vert(__global uchar *res, int res_step, int res_offset, int res_rows, int res_cols,
                             __global uchar *adist, int adist_step, int adist_offset,
                             int write_new_a_dist)

{
    int j = get_global_id(0);

    if (!(j >= 0 && j < res_cols))
        return;

    res   += mad24(j, (int)ELEM_SIZE, res_offset);
    adist += mad24(j, (int)sizeof(float), adist_offset);

    SrcVec cur_val;
    float ad;
    int i;

    cur_val = getPix(res + 0*res_step, 0);
    for (i = 1; i < res_rows; i++)
    {
        ad = getFloat(adist + (i-1)*adist_step, 0);
        cur_val = (1.0f - ad)*getPix(res + i*res_step, 0) + ad*cur_val;
        storePix(cur_val, res + i*res_step, 0);
    }

    for (i = res_rows - 2; i >= 0; i--)
    {
        ad = getFloat(adist + i*adist_step, 0);
        cur_val = (1.0f - ad)*getPix(res + i*res_step, 0) + ad*cur_val;
        storePix(cur_val, res + i*res_step, 0);

        if (write_new_a_dist)
            storeFloat(ad*ad, adist + i*adist_step, 0);
    }
}

#define genMinMaxPtrs(ptr, h) __global const uchar *ptr##_min = (__global const uchar *)ptr;\
                              __global const uchar *ptr##_max = (__global const uchar *)ptr + mad24(h, ptr##_step, ptr##_offset);

#define checkPtr(ptr, bound) (bound##_min <= (__global const uchar *)(ptr) && (__global const uchar *)(ptr) < bound##_max)

__kernel void filter_RF_block_init_fwd(__global uchar *res, int res_step, int res_offset, int rows, int cols,
                                       __global uchar *adist, int adist_step, int adist_offset,
                                       __global uchar *weights, int weights_step, int weights_offset,
                                       int blockSize)

{
    int bid = get_global_id(0);
    int j = get_global_id(1);
    if (j < 0 || j >= cols) return;

    int startRow = max(1, bid*blockSize); //skip first row
    int endRow = min(rows, (bid + 1)*blockSize);

    res     += mad24(j, ELEM_SIZE, mad24(startRow, res_step, res_offset));;
    adist   += mad24(j, (int) sizeof(float), mad24(startRow-1, adist_step, adist_offset));
    weights += mad24(j, (int) sizeof(float), mad24(startRow, weights_step, weights_offset));

    SrcVec cur_val = (startRow != 1) ? 0 : getPix(res - res_step, 0);
    float weight0 = 1.0f;

    for (int i = startRow; i < endRow; i++)
    {
        float ad = getFloat(adist, 0);

        cur_val = (1.0f - ad)*getPix(res, 0) + ad*cur_val;
        storePix(cur_val, res, 0);
        
        weight0 *= ad;
        storeFloat(weight0, weights, 0);        

        res += res_step;
        adist += adist_step;
        weights += weights_step;
    }
}

__kernel void filter_RF_block_fill_borders_fwd(__global uchar *res, int res_step, int res_offset, int rows, int cols,
                                               __global uchar *weights, int weights_step, int weights_offset,
                                               int blockSize)
{
    int j = get_global_id(0);
    if (j < 0 || j >= cols) return;

    int startRow = 2*blockSize - 1;
    res     += mad24(j, ELEM_SIZE, mad24(startRow, res_step, res_offset));
    weights += mad24(j, (int) sizeof(float), mad24(startRow, weights_step, weights_offset));

    int res_step_block = blockSize*res_step;
    int weights_step_block = blockSize*weights_step;
    SrcVec prev_pix = getPix(res - res_step_block, 0);

    for (int i = startRow; i < rows; i += blockSize)
    {
        prev_pix = getPix(res, 0) + getFloat(weights, 0)*prev_pix;
        storePix(prev_pix, res, 0);

        res += res_step_block;
        weights += weights_step_block;
    }
}

__kernel void filter_RF_block_fill_fwd(__global uchar *res, int res_step, int res_offset, int rows, int cols,
                                       __global uchar *weights, int weights_step, int weights_offset,
                                       int blockSize)
{
    int i = get_global_id(0) + blockSize;
    int j = get_global_id(1);
    if (j < 0 || j >= cols || i < 0 || i >= rows) return;
    if (i % blockSize == blockSize - 1) return; //to avoid rewriting bound pixels

    int bid = i / blockSize;
    int ref_row_id = bid*blockSize - 1;

    __global uchar *res_ref_row = res + mad24(ref_row_id, res_step, res_offset);
    __global uchar *res_dst_row = res + mad24(i, res_step, res_offset);
    __global float *weights_row = PtrAdd(weights, mad24(i, weights_step, weights_offset), float);

    storePix(getPix(res_dst_row, j) + weights_row[j]*getPix(res_ref_row, j), res_dst_row, j);
}

__kernel void filter_RF_block_init_bwd(__global uchar *res, int res_step, int res_offset, int rows, int cols,
                                       __global uchar *adist, int adist_step, int adist_offset,
                                       __global uchar *weights, int weights_step, int weights_offset,
                                       int blockSize)
{
    int bid = get_global_id(0);
    int j = get_global_id(1);
    if (j < 0 || j >= cols) return;

    int startRow = rows-1 - max(1, bid*blockSize);
    int endRow = max(-1, rows-1 - (bid+1)*blockSize);

    res     += mad24(j, ELEM_SIZE, mad24(startRow, res_step, res_offset));;
    adist   += mad24(j, (int) sizeof(float), mad24(startRow, adist_step, adist_offset));
    weights += mad24(j, (int) sizeof(float), mad24(startRow, weights_step, weights_offset));

    SrcVec cur_val = (startRow != rows-2) ? 0 : getPix(res + res_step, 0);
    float weight0 = 1.0f;
    
    for (int i = startRow; i > endRow; i--)
    {
        float ad = getFloat(adist, 0);

        cur_val = (1.0f - ad)*getPix(res, 0) + ad*cur_val;
        storePix(cur_val, res, 0);
        
        weight0 *= ad;
        storeFloat(weight0, weights, 0);

        res -= res_step;
        adist -= adist_step;
        weights -= weights_step;
    }
}

__kernel void filter_RF_block_fill_borders_bwd(__global uchar *res, int res_step, int res_offset, int rows, int cols,
                                               __global uchar *weights, int weights_step, int weights_offset,
                                               int blockSize)
{
    int j = get_global_id(0);
    if (j < 0 || j >= cols) return;

    int startRow = rows-1 - (2*blockSize - 1);
    res     += mad24(j, ELEM_SIZE, mad24(startRow, res_step, res_offset));
    weights += mad24(j, (int) sizeof(float), mad24(startRow, weights_step, weights_offset));

    int res_step_block = blockSize*res_step;
    int weights_step_block = blockSize*weights_step;
    SrcVec prev_pix = getPix(res + res_step_block, 0);

    for (int i = startRow; i >= 0; i -= blockSize)
    {
        prev_pix = getPix(res, 0) + getFloat(weights, 0)*prev_pix;
        storePix(prev_pix, res, 0);

        res -= blockSize*res_step;
        weights -= blockSize*weights_step;
    }
}

__kernel void filter_RF_block_fill_bwd(__global uchar *res, int res_step, int res_offset, int rows, int cols,
                                       __global uchar *weights, int weights_step, int weights_offset,
                                       int blockSize)
{
    int i = get_global_id(0) + blockSize;
    int j = get_global_id(1);
    if (j < 0 || j >= cols || i < 0 || i >= rows) return;
    if (i % blockSize == blockSize - 1) return; //to avoid rewriting bound pixels

    int bid = i / blockSize;
    int ref_row_id = rows-1 - (bid*blockSize - 1);
    int dst_row_id = rows-1 - i;

    __global uchar *res_ref_row = res + mad24(ref_row_id, res_step, res_offset);
    __global uchar *res_dst_row = res + mad24(dst_row_id, res_step, res_offset);
    __global float *weights_row = PtrAdd(weights, mad24(dst_row_id, weights_step, weights_offset), float);
    
    storePix(getPix(res_dst_row, j) + weights_row[j]*getPix(res_ref_row, j), res_dst_row, j);
}