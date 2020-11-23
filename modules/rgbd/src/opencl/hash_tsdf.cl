// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

//#define NAN_NUM -2147483648;

typedef __INT8_TYPE__ int8_t;
typedef __UINT32_TYPE__ uint32_t;
typedef __INT32_TYPE__ int32_t;

typedef int8_t TsdfType;
typedef uchar WeightType;

struct TsdfVoxel
{
    TsdfType tsdf;
    WeightType weight;
};

struct Volume_NODE
{
    int4 idx;
    int32_t row;
    int32_t nextVolumeRow;
    //bool isActive;
    //int lastVisibleIndex;
    //float16 vol2camMatrix;
    int32_t tmp;
};

static inline TsdfType floatToTsdf(float num)
{
    int8_t res = (int8_t) ( (num * (-128)) );
    res = res ? res : (num < 0 ? 1 : -1);
    return res;
}

static inline float tsdfToFloat(TsdfType num)
{
    return ( (float) num ) / (-128);
}

__kernel void preCalculationPixNorm (__global float * pixNorms,
                                     const __global float * xx,
                                     const __global float * yy,
                                     int width)
{    
    int i = get_global_id(0);
    int j = get_global_id(1);
    int idx = i*width + j;
    pixNorms[idx] = sqrt(xx[j] * xx[j] + yy[i] * yy[i] + 1.0f);
}

uint calc_hash(int3 x)
{
    uint32_t seed = 0;
    //uint GOLDEN_RATIO = 0x9e3779b9;
    uint32_t GOLDEN_RATIO = 0x9e3779b9;
    seed ^= x.x + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
    seed ^= x.y + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
    seed ^= x.z + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
    return seed;
}

int findVolume(__global struct Volume_NODE * hash_table, int3 indx,
               int list_size, int bufferNums, int hash_divisor)
{
    int hash = calc_hash(indx) % hash_divisor;
    printf("hash = %d \n", hash);
    
    //printf("%d | %d \n", calc_hash(indx), hash);
    int num = 1;
    int i = hash * num * list_size;
    int NAN_NUM = -2147483648;
    while (i != NAN_NUM)
    {
        __global struct Volume_NODE* v = (hash_table + i);
        printf(" = %d %d %d \n", v->idx.x, v->idx.y, v->idx.z);
        printf("   %d | %d \n", v->row , v->nextVolumeRow);
        //int3 tmp = v->idx 
        if (v->idx.x == indx.x &&
            v->idx.y == indx.y &&
            v->idx.z == indx.z)
            return v->row;
        if (v->idx.x == NAN_NUM)
            return -2;
        i = v->nextVolumeRow;
    }

    return 1;
}

__kernel void integrateAllVolumeUnits(
                        __global const char * depthptr,
                        int depth_step, int depth_offset,
                        int depth_rows, int depth_cols,
                        __global struct Volume_NODE * hash_table,
                        const int list_size, 
                        const int bufferNums, 
                        const int hash_divisor,
                        // const float16 vol2camMatrix,
                        // const __global struct TsdfVoxel * volUnitsData,
                        //global bool * isActive,
                        // const __global int * lastVisibleIndexes,
                        // const __global float * pixNorms,
                        const float voxelSize,
                        const int4 volResolution4,
                        const int4 volDims4,
                        const float2 fxy,
                        const float2 cxy,
                        const float dfac,
                        const float truncDist,
                        const int maxWeight
                        // ,const __global float * pixNorms
                        )
{

    //int i = get_global_id(0);
    //int j = get_global_id(1);
    //int k = get_global_id(2);
    //printf("[i, j, k] = [%d , %d , %d] \n", i, j, k);

    //printf(" = %d , %d , %d \n", hash_params.x, hash_params.y, hash_params.z);
    //printf(" = %d , %d , %d \n", list_size, bufferNums, hash_divisor );
    //int tmp = findVolume(hash_table, hash_table->idx, hash_divisor);
    
    //int3 idx; idx.x = 7; idx.y=7;idx.z=1;
    //int tmp = findVolume(hash_table, idx, list_size, bufferNums, hash_divisor);
    //printf("idx = %d %d %d | row = %d \n", idx.x, idx.y, idx.z, tmp);
    
    //hash_table->idx.x = 10;
    //findVolume(hash_table, hash_table->idx, hash_divisor);
    
    //printf("\n");
    
    //char* tmp = (char) hash_table;
    //for (int i = 0; i < 100; i++)
    //{
    //    printf("|%c", tmp[i] );
    //}

    
    for (int i = 0; i < 10; i++)
    {
        printf("\n");
        int idx = i;
        struct Volume_NODE v = hash_table[idx];
        printf("idx = %d %d %d \n", v.idx.x, v.idx.y, v.idx.z);
        printf("row = %d \n", v.row);
        printf("nv  = %d \n", v.nextVolumeRow);
        printf("tmp = %d \n", v.tmp);

        //printf("idx = %d %d %d \n", (hash_table[idx]).idx.x, (hash_table[idx]).idx.y, (hash_table[idx]).idx.z);
        //printf("idx = %d %d %d \n", (hash_table+idx)->idx.x, (hash_table+idx)->idx.y, (hash_table+idx)->idx.z);
        
    }
    
    //printf("idx = %d %d %d \n", hash_table->idx.x, hash_table->idx.y, hash_table->idx.z);
    //printf("row = %d \n", hash_table->row);
    //printf("nv  = %d \n", hash_table->nextVolumeRow);
    //printf("isA = %d \n", hash_table->isActive);
    //printf("isA = %s \n", hash_table->isActive ? "true" : "false");
    //printf("lvi = %d \n", hash_table->lastVisibleIndex);
    //printf("%f %f \n", hash_table->vol2camMatrix.x, hash_table->vol2camMatrix.y);

}

__kernel void integrateVolumeUnit(__global const char * depthptr,
                        int depth_step, int depth_offset,
                        int depth_rows, int depth_cols,
                        __global struct TsdfVoxel * volumeptr,
                        const float16 vol2camMatrix,
                        const float voxelSize,
                        const int4 volResolution4,
                        const int4 volDims4,
                        const float2 fxy,
                        const float2 cxy,
                        const float dfac,
                        const float truncDist,
                        const int maxWeight,
                        const __global float * pixNorms)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    const int3 volResolution = volResolution4.xyz;

    if(x >= volResolution.x || y >= volResolution.y)
        return;

    // coord-independent constants
    const int3 volDims = volDims4.xyz;
    const float2 limits = (float2)(depth_cols-1, depth_rows-1);

    const float4 vol2cam0 = vol2camMatrix.s0123;
    const float4 vol2cam1 = vol2camMatrix.s4567;
    const float4 vol2cam2 = vol2camMatrix.s89ab;

    const float truncDistInv = 1.f/truncDist;

    // optimization of camSpace transformation (vector addition instead of matmul at each z)
    float4 inPt = (float4)(x*voxelSize, y*voxelSize, 0, 1);
    float3 basePt = (float3)(dot(vol2cam0, inPt),
                             dot(vol2cam1, inPt),
                             dot(vol2cam2, inPt));

    float3 camSpacePt = basePt;

    // zStep == vol2cam*(float3(x, y, 1)*voxelSize) - basePt;
    float3 zStep = ((float3)(vol2cam0.z, vol2cam1.z, vol2cam2.z))*voxelSize;

    int volYidx = x*volDims.x + y*volDims.y;

    int startZ, endZ;
    if(fabs(zStep.z) > 1e-5)
    {
        int baseZ = convert_int(-basePt.z / zStep.z);
        if(zStep.z > 0)
        {
            startZ = baseZ;
            endZ = volResolution.z;
        }
        else
        {
            startZ = 0;
            endZ = baseZ;
        }
    }
    else
    {
        if(basePt.z > 0)
        {
            startZ = 0; endZ = volResolution.z;
        }
        else
        {
            // z loop shouldn't be performed
            //startZ = endZ = 0;
            return;
        }
    }

    startZ = max(0, startZ);
    endZ = min(volResolution.z, endZ);

    for(int z = startZ; z < endZ; z++)
    {
        // optimization of the following:
        //float3 camSpacePt = vol2cam * ((float3)(x, y, z)*voxelSize);
        camSpacePt += zStep;

        if(camSpacePt.z <= 0)
            continue;

        float3 camPixVec = camSpacePt / camSpacePt.z;
        float2 projected = mad(camPixVec.xy, fxy, cxy);

        float v;
        // bilinearly interpolate depth at projected
        if(all(projected >= 0) && all(projected < limits))
        {
            float2 ip = floor(projected);
            int xi = ip.x, yi = ip.y;

            __global const float* row0 = (__global const float*)(depthptr + depth_offset +
                                                                 (yi+0)*depth_step);
            __global const float* row1 = (__global const float*)(depthptr + depth_offset +
                                                                 (yi+1)*depth_step);

            float v00 = row0[xi+0];
            float v01 = row0[xi+1];
            float v10 = row1[xi+0];
            float v11 = row1[xi+1];
            float4 vv = (float4)(v00, v01, v10, v11);

            // assume correct depth is positive
            if(all(vv > 0))
            {
                float2 t = projected - ip;
                float2 vf = mix(vv.xz, vv.yw, t.x);
                v = mix(vf.s0, vf.s1, t.y);
            }
            else
                continue;
        }
        else
            continue;

        if(v == 0)
            continue;

        int idx = projected.y * depth_rows + projected.x;
        float pixNorm = pixNorms[idx];
        //float pixNorm = length(camPixVec);

        // difference between distances of point and of surface to camera
        float sdf = pixNorm*(v*dfac - camSpacePt.z);
        // possible alternative is:
        // float sdf = length(camSpacePt)*(v*dfac/camSpacePt.z - 1.0);

        if(sdf >= -truncDist)
        {
            float tsdf = fmin(1.0f, sdf * truncDistInv);
            int volIdx = volYidx + z*volDims.z;

            struct TsdfVoxel voxel = volumeptr[volIdx];
            float value  = tsdfToFloat(voxel.tsdf);
            int weight = voxel.weight;

            // update TSDF
            value = (value*weight + tsdf) / (weight + 1);
            weight = min(weight + 1, maxWeight);

            voxel.tsdf = floatToTsdf(value);
            voxel.weight = weight;
            volumeptr[volIdx] = voxel;
        }
    }
}