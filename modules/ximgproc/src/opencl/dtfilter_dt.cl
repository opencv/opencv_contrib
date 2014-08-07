#if defined(_MSC_VER) //for correct syntax highlighting
#include "OpenCLKernel.hpp"
#define cn 3
#define GuideType uchar
#define GuideVec uchar3
#define convert_guide convert_float3
#define NUM_ITERS 3
#endif

#if !defined(convert_guide)
#define convert_guide(a) (a)
#endif

#if cn==3
#define ELEM_SIZE (3*sizeof(GuideType))
#define loadGuide(addr) vload3(0, (__global const GuideType *)(addr))
#else
#define ELEM_SIZE sizeof(GuideVec)
#define loadGuide(addr) ( *(__global const GuideVec*)(addr) ) 
#endif

#define storeDist(val, addr) *(__global float*)(addr) = val

#if cn == 1
#define SUM(a) (a)
#elif cn == 2
#define SUM(a) (a.x + a.y)
#elif cn == 3
#define SUM(a) (a.x + a.y + a.z)
#elif cn == 4
#define SUM(a) (a.x + a.y + a.z + a.w)
#else
#error "cn should be <= 4"
#endif

#define NORM(a, b) SUM(fabs(convert_guide(a) - convert_guide(b)))
#define DT(a, b, sigmaRatios) (1.0f + (sigmaRatios)*NORM(a, b))

#define getFloat(addr, index)           ( *(__global const float*)(addr + (index)*sizeof(float)) )
#define storeFloat(val, addr, index)    ( *(__global float*)(addr + (index)*sizeof(float)) = val )

#define PtrAdd(addr, bytesNum, Type)        ((__global Type *)((__global uchar*)(addr) + bytesNum))
#define PtrAddConst(addr, bytesNum, Type)   ((__global const Type *)((__global const uchar*)(addr) + bytesNum))

__kernel void find_conv_bounds_by_idt(__global const uchar *idist, int idist_step, int idist_offset, int rows, int cols,
                                      __global       uchar *bounds, int bounds_step, int bounds_offset, 
                                      float radius1, float radius2, float radius3)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (!(i >= 0 && i < rows) || !(j >= 0 && j < cols))
        return;

    idist += mad24(i, idist_step, idist_offset);
    __global int *bound = bounds + mad24((NUM_ITERS-1)*rows + i, bounds_step, mad24(j, 2*(int)sizeof(int), bounds_offset));

    float center_idt_val = getFloat(idist, j);
    float radius[] = {radius1, radius2, radius3};

    int low_bound = j;
    int high_bound = j;
    float search_idt_val;
    for (int iter = NUM_ITERS - 1; iter >= 0; iter--)
    {
        search_idt_val = center_idt_val - radius[iter];
        while (search_idt_val < getFloat(idist, low_bound-1))
            low_bound--;
        bound[0] = low_bound;

        search_idt_val = center_idt_val + radius[iter];
        while (getFloat(idist, high_bound + 1) < search_idt_val)
            high_bound++;
        bound[1] = high_bound;

        bound = PtrAdd(bound, -rows*bounds_step, int);
    }
}

__kernel void find_conv_bounds_by_dt(__global const uchar *dist, int dist_step, int dist_offset, int rows, int cols,
                                     __global       uchar *bounds, int bounds_step, int bounds_offset,
                                     float radius1, float radius2, float radius3)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (!(i >= 0 && i < rows) || !(j >= 0 && j < cols))
        return;

    dist += mad24(i, dist_step, dist_offset);
    int rowsOffset = mad24(NUM_ITERS - 1, rows, i);
    __global int *bound = PtrAdd(bounds, mad24(rowsOffset, bounds_step, mad24(j, 2*(int)sizeof(int), bounds_offset)), int);

    float radius[] = {radius1, radius2, radius3};

    int low_bound = j;
    int high_bound = j;
    float val, cur_radius, low_dt_val = 0.0f, high_dt_val = 0.0f;
    for (int iter = NUM_ITERS - 1; iter >= 0; iter--)
    {
        cur_radius = radius[iter];

        while (cur_radius > (val = low_dt_val + getFloat(dist, low_bound - 1)) )
        {
            low_dt_val = val;
            low_bound--;
        }
        bound[0] = low_bound;

        while (cur_radius > (val = high_dt_val + getFloat(dist, high_bound)) )
        {
            high_dt_val = val;
            high_bound++;
        }
        bound[1] = high_bound;

        bound = PtrAdd(bound, -rows*bounds_step, int);
    }
}

__kernel void find_conv_bounds_and_tails(__global const uchar *dist, int dist_step, int dist_offset, int rows, int cols,
                                         __global uchar *bounds, int bounds_step, int bounds_offset, 
                                         __global float *tailmat, int tail_step, int tail_offset,
                                         float radius1, float radius2, float radius3)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (!(i >= 0 && i < rows) || !(j >= 0 && j < cols))
        return;

    dist += mad24(i, dist_step, dist_offset);

    int rowsOffset = mad24(NUM_ITERS - 1, rows, i);
    __global int *bound = PtrAdd(bounds, mad24(rowsOffset, bounds_step, mad24(j, 2*(int)sizeof(int), bounds_offset)), int);
    __global float *tail = PtrAdd(tailmat, mad24(rowsOffset, tail_step, mad24(j, 2*(int)sizeof(float), tail_offset)), float);

    float radius[] = {radius1, radius2, radius3};

    int low_bound = j;
    int high_bound = j;
    float val, cur_radius, low_dt_val = 0.0f, high_dt_val = 0.0f;
    for (int iter = NUM_ITERS - 1; iter >= 0; iter--)
    {
        cur_radius = radius[iter];

        while (cur_radius > (val = low_dt_val + getFloat(dist, low_bound - 1)) )
        {
            low_dt_val = val;
            low_bound--;
        }
        bound[0] = low_bound;
        tail[0] = (cur_radius - low_dt_val);

        while (cur_radius > (val = high_dt_val + getFloat(dist, high_bound)) )
        {
            high_dt_val = val;
            high_bound++;
        }
        bound[1] = high_bound;
        tail[1] = (cur_radius - high_dt_val);

        bound = PtrAdd(bound, -rows*bounds_step, int);
        tail = PtrAdd(tail, -rows*tail_step, float);
    }
}

__kernel void compute_dt_hor(__global const uchar *src, int src_step, int src_offset, int src_rows, int src_cols,
                             __global       uchar *dst, int dst_step, int dst_offset,
                             float sigma_ratio, float max_radius)
{
    int i = get_global_id(0);
    int j = get_global_id(1) - 1;

    if (!(i >= 0 && i < src_rows && j >= -1 && j < src_cols))
        return;

    src += mad24(i, src_step, mad24(j, (int)ELEM_SIZE, src_offset));
    dst += mad24(i, dst_step, mad24(j, (int)sizeof(float) , dst_offset));

    float dist;
    if (j == -1 || j == src_cols - 1)
        dist = max_radius;
    else
        dist = DT(loadGuide(src), loadGuide(src + ELEM_SIZE), sigma_ratio);
    storeDist(dist, dst);
}

__kernel void compute_dt_vert(__global const uchar *src, int src_step, int src_offset, int src_rows, int src_cols,
                              __global       uchar *dst, int dst_step, int dst_offset,
                              float sigma_ratio, float max_radius)
{
    int i = get_global_id(0) - 1;
    int j = get_global_id(1);

    if (!(i >= -1 && i < src_rows && j >= 0 && j < src_cols))
        return;

    src += mad24(i, src_step, mad24(j, (int)ELEM_SIZE, src_offset));
    dst += mad24(i, dst_step, mad24(j, (int)sizeof(float) , dst_offset));

    float dist;
    if (i == -1 || i == src_rows - 1)
        dist = max_radius;
    else
        dist = DT(loadGuide(src), loadGuide(src + src_step), sigma_ratio);
    storeDist(dist, dst);
}

__kernel void compute_idt_vert(__global const uchar *src, int src_step, int src_offset, int src_rows, int src_cols,
                               __global       uchar *dst, int dst_step, int dst_offset,
                               float sigma_ratio)
{
    int j = get_global_id(0);
    
    if (!(j >= 0 && j < src_cols))
        return;
    
    int i;
    float idist = 0;
        
    __global const uchar *src_col = src + mad24(j, (int)ELEM_SIZE, src_offset);
    __global       uchar *dst_col = dst + mad24(j, (int)sizeof(float), dst_offset);
        
    storeFloat(-FLT_MAX, dst_col + (-1)*dst_step, 0);
    storeFloat(0.0f, dst_col + 0*dst_step, 0);
    for (i = 1; i < src_rows; i++, src_col += src_step)
    {
        idist += DT(loadGuide(src_col), loadGuide(src_col + src_step), sigma_ratio);
        storeFloat(idist, dst_col + i*dst_step, 0);
    }
    storeFloat(FLT_MAX, dst_col + i*dst_step, 0);
}

__kernel void compute_a0DT_vert(__global const uchar *src, int src_step, int src_offset, int src_rows, int src_cols,
                                __global       uchar *dst, int dst_step, int dst_offset,
                                float sigma_ratio, float alpha)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (!(i >= 0 && i < src_rows-1 && j >= 0 && j < src_cols))
        return;

    src += mad24(i, src_step, mad24(j, (int)ELEM_SIZE, src_offset));
    dst += mad24(i, dst_step, mad24(j, (int)sizeof(float) , dst_offset));

    float dist = DT(loadGuide(src), loadGuide(src + src_step), sigma_ratio);
    storeDist(native_powr(alpha, dist), dst);
}