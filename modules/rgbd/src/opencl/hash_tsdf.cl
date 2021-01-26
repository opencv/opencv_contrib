// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#define NAN_ELEMENT -2147483647
#define USE_INTERPOLATION_IN_GETNORMAL 1

typedef __INT8_TYPE__ int8_t;
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
    int32_t dummy;
    int32_t dummy2;
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

static uint calc_hash(int3 x)
{
    unsigned int seed = 0;
    unsigned int GOLDEN_RATIO = 0x9e3779b9;
    seed ^= x.s0 + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
    seed ^= x.s1 + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
    seed ^= x.s2 + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
    return seed;
}


static int findRow(__global struct Volume_NODE * hash_table, int3 indx,
                   int list_size, int bufferNums, int hash_divisor)
{
    int bufferNum = 0;
    int hash = calc_hash(indx) % hash_divisor;
    int i = (bufferNum * hash_divisor + hash) * list_size;
    while (i >= 0)
    {
        struct Volume_NODE v = hash_table[i];
        if (all(v.idx.s012 == indx.s012))
            return v.row;
        else
            i = v.nextVolumeRow;
    }

    return -1;
}


static void integrateVolumeUnit(
                        int x, int y,
                        __global const char * depthptr,
                        int depth_step, int depth_offset,
                        int depth_rows, int depth_cols,
                        __global struct TsdfVoxel * volumeptr,
                        const __global float * pixNorms,
                        const float16 vol2camMatrix,
                        const float voxelSize,
                        const int4 volResolution4,
                        const int4 volDims4,
                        const float2 fxy,
                        const float2 cxy,
                        const float dfac,
                        const float truncDist,
                        const int maxWeight
                        )
{
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
            startZ = endZ = 0;
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
        float2 projected = mad(camPixVec.xy, fxy, cxy); // mad(a,b,c) = a * b + c

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

__kernel void integrateAllVolumeUnits(
                        __global const char * depthptr,
                        int depth_step, int depth_offset,
                        int depth_rows, int depth_cols,
                        __global const int4 * volumeUnitIndices,
                        int volumeUnitIndices_step, int volumeUnitIndices_offset,
                        int volumeUnitIndices_rows, int volumeUnitIndices_cols,
                        __global struct TsdfVoxel * allVolumePtr,
                        int table_step, int table_offset,
                        int table_rows, int table_cols,
                        __global const float * pixNorms,
                        __global const float * allVol2camMatrix,
                        int val2cam_step, int val2cam_offset,
                        int val2cam_rows, int val2cam_cols,
                        __global const uchar* isActiveFlagsPtr,
                        int isActiveFlagsStep, int isActiveFlagsOffset,
                        int isActiveFlagsRows, int isActiveFlagsCols,
                        const float voxelSize,
                        const int4 volResolution4,
                        const int4 volDims4,
                        const float2 fxy,
                        const float2 cxy,
                        const float dfac,
                        const float truncDist,
                        const int maxWeight
                        )
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int row = get_global_id(2);
    int4 idx = volumeUnitIndices[row];

    int isActive = (__global const uchar*)(isActiveFlagsPtr + isActiveFlagsOffset + (row));

    if (isActive)
    {
        __global struct TsdfVoxel * volumeptr = (__global struct TsdfVoxel*)
                                                (allVolumePtr + table_offset + (row) * 16*16*16);
        __global const float * p_vol2camMatrix = (__global const float *)
                                                 (allVol2camMatrix + val2cam_offset + (row) * 16);
        
        const float16 vol2camMatrix = vload16(0, p_vol2camMatrix);

        integrateVolumeUnit(
            i, j,
            depthptr,
            depth_step, depth_offset,
            depth_rows, depth_cols,
            volumeptr,
            pixNorms,
            vol2camMatrix,
            voxelSize,
            volResolution4,
            volDims4,
            fxy,
            cxy,
            dfac,
            truncDist,
            maxWeight
            );
    }
}

static struct TsdfVoxel at(int3 volumeIdx, int row, int volumeUnitResolution,
                           int3 volStrides, __global struct TsdfVoxel * allVolumePtr, int table_offset)

{
    //! Out of bounds
    if (any(volumeIdx >= volumeUnitResolution) ||
        any(volumeIdx < 0))

    {
        struct TsdfVoxel dummy;
        dummy.tsdf = floatToTsdf(1.0f);
        dummy.weight = 0;
        return dummy;
    }

    __global struct TsdfVoxel * volData = (__global struct TsdfVoxel*)
                                          (allVolumePtr + table_offset + row * 16*16*16);
    int3 ismul = volumeIdx * volStrides;
    int coordBase = ismul.x + ismul.y + ismul.z;
    return volData[coordBase];
}


static struct TsdfVoxel atVolumeUnit(int3 volumeIdx, int3 volumeUnitIdx, int row,
                                     int volumeUnitResolution, int3 volStrides,
                                     __global const struct TsdfVoxel * allVolumePtr, int table_offset)

{
    //! Out of bounds
    if (row < 0)
    {
        struct TsdfVoxel dummy;
        dummy.tsdf = floatToTsdf(1.0f);
        dummy.weight = 0;
        return dummy;
    }

    int3 volUnitLocalIdx = volumeIdx - volumeUnitIdx * volumeUnitResolution;
    __global struct TsdfVoxel * volData = (__global struct TsdfVoxel*)
                                          (allVolumePtr + table_offset + row * 16*16*16);
    int3 ismul = volUnitLocalIdx * volStrides;
    int coordBase = ismul.x + ismul.y + ismul.z;
    return volData[coordBase];
}

inline float interpolate(float tx, float ty, float tz, float vx[8])
{
    float v00 = vx[0] + tz * (vx[1] - vx[0]);
    float v01 = vx[2] + tz * (vx[3] - vx[2]);
    float v10 = vx[4] + tz * (vx[5] - vx[4]);
    float v11 = vx[6] + tz * (vx[7] - vx[6]);

    float v0 = v00 + ty * (v01 - v00);
    float v1 = v10 + ty * (v11 - v10);

    return v0 + tx * (v1 - v0);
}

inline float3 getNormalVoxel(float3 p, __global const struct TsdfVoxel* allVolumePtr,
                             int3 volResolution,
                             float voxelSizeInv,
                             __global struct Volume_NODE * hash_table,
                             const int list_size,
                             const int bufferNums,
                             const int hash_divisor,
                             int3 volStrides, int table_offset)
{
    
    float3 normal = (float3) (0.0f, 0.0f, 0.0f);
    float3 ptVox = p * voxelSizeInv;
    int3 iptVox = convert_int3(floor(ptVox));

    // A small hash table to reduce a number of findRow() calls
    // -2 and lower means not queried yet
    // -1 means not found
    // 0+ means found
    int  iterMap[8];
    for (int i = 0; i < 8; i++)
    {
        iterMap[i] = -2;
    }

#if !USE_INTERPOLATION_IN_GETNORMAL
    int3 offsets[] = { { 1,  0,  0}, {-1,  0,  0}, { 0,  1,  0}, // 0-3
                       { 0, -1,  0}, { 0,  0,  1}, { 0,  0, -1}  // 4-7
    };
    
    const int nVals = 6;
    float vals[6];
#else
    int3 offsets[]={{ 0,  0,  0}, { 0,  0,  1}, { 0,  1,  0}, { 0,  1,  1}, //  0-3
                    { 1,  0,  0}, { 1,  0,  1}, { 1,  1,  0}, { 1,  1,  1}, //  4-7
                    {-1,  0,  0}, {-1,  0,  1}, {-1,  1,  0}, {-1,  1,  1}, //  8-11
                    { 2,  0,  0}, { 2,  0,  1}, { 2,  1,  0}, { 2,  1,  1}, // 12-15
                    { 0, -1,  0}, { 0, -1,  1}, { 1, -1,  0}, { 1, -1,  1}, // 16-19
                    { 0,  2,  0}, { 0,  2,  1}, { 1,  2,  0}, { 1,  2,  1}, // 20-23
                    { 0,  0, -1}, { 0,  1, -1}, { 1,  0, -1}, { 1,  1, -1}, // 24-27
                    { 0,  0,  2}, { 0,  1,  2}, { 1,  0,  2}, { 1,  1,  2}, // 28-31
    };
    const int nVals = 32;
    float vals[32];
#endif

    for (int i = 0; i < nVals; i++)
    {
        int3 pt = iptVox + offsets[i];

        // VoxelToVolumeUnitIdx() 
        // TODO: add assertion - if (!(vuRes & (vuRes - 1)))
        int3 volumeUnitIdx = convert_int3(floor(convert_float3(pt.s012) / convert_float3(volResolution.s012)));

        int3 vand = (volumeUnitIdx & 1);
        int dictIdx = vand.s0 + vand.s1 * 2 + vand.s2 * 4;

        int it = iterMap[dictIdx];
        if (it < -1)
        {
            it = findRow(hash_table, volumeUnitIdx, list_size, bufferNums, hash_divisor);
            iterMap[dictIdx] = it;
        }

        struct TsdfVoxel tmp = atVolumeUnit(pt, volumeUnitIdx, it, volResolution.s0, volStrides, allVolumePtr, table_offset);
        vals[i] = tsdfToFloat( tmp.tsdf );
    }

#if !USE_INTERPOLATION_IN_GETNORMAL
    normal.s0 = vals[0 * 2] - vals[0 * 2 + 1];
    normal.s1 = vals[1 * 2] - vals[1 * 2 + 1];
    normal.s2 = vals[2 * 2] - vals[2 * 2 + 1];
#else

    float cxv[8], cyv[8], czv[8];

    // How these numbers were obtained:
    // 1. Take the basic interpolation sequence:
    // 000, 001, 010, 011, 100, 101, 110, 111
    // where each digit corresponds to shift by x, y, z axis respectively.
    // 2. Add +1 for next or -1 for prev to each coordinate to corresponding axis
    // 3. Search corresponding values in offsets
    const int idxxp[8] = { 8,  9, 10, 11,  0,  1,  2,  3 };
    const int idxxn[8] = { 4,  5,  6,  7, 12, 13, 14, 15 };
    const int idxyp[8] = { 16, 17,  0,  1, 18, 19,  4,  5 };
    const int idxyn[8] = { 2,  3, 20, 21,  6,  7, 22, 23 };
    const int idxzp[8] = { 24,  0, 25,  2, 26,  4, 27,  6 };
    const int idxzn[8] = { 1, 28,  3, 29,  5, 30,  7, 31 };

    for (int i = 0; i < 8; i++)
    {
        cxv[i] = vals[idxxn[i]] - vals[idxxp[i]];
        cyv[i] = vals[idxyn[i]] - vals[idxyp[i]];
        czv[i] = vals[idxzn[i]] - vals[idxzp[i]];
    }

    float tx = ptVox.x - iptVox.x;
    float ty = ptVox.y - iptVox.y;
    float tz = ptVox.z - iptVox.z;

    normal.s0 = interpolate(tx, ty, tz, cxv);
    normal.s1 = interpolate(tx, ty, tz, cyv);
    normal.s2 = interpolate(tx, ty, tz, czv);
    
    //if(!any(isnan(normal)))
    //    printf("%f, %f, %f" ,normal.s0, normal.s1, normal.s2);

#endif

    float norm = sqrt(dot(normal, normal));
    return norm < 0.0001f ? nan((uint)0) : normal / norm;
}

typedef float4 ptype;

__kernel void raycast(
                    __global struct Volume_NODE * hash_table,
                    const int list_size, 
                    const int bufferNums, 
                    const int hash_divisor,
                    const int lastVolIndex, 
                    __global char * pointsptr,
                      int points_step, int points_offset,
                    __global char * normalsptr,
                      int normals_step, int normals_offset,
                    const int2 frameSize,
                    __global struct TsdfVoxel * allVolumePtr,
                        int table_step, int table_offset,
                        int table_rows, int table_cols,
                    float4  cam2volTransGPU,
                    float16 cam2volRotGPU,
                    float16 vol2camRotGPU,
                    float truncateThreshold,
                    const float2 fixy, const float2 cxy,
                    const float4 boxDown4, const float4 boxUp4,
                    const float tstep,
                    const float voxelSize,
                    const int4 volResolution4,
                    const int4 volDims4,
                    const int8 neighbourCoords,
                    float voxelSizeInv,
                    float volumeUnitSize,
                    float truncDist,
                    int volumeUnitResolution,
                    int4 volStrides4
                    )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= frameSize.x || y >= frameSize.y)
        return;

    float3 point  = nan((uint)0);
    float3 normal = nan((uint)0);

    const float3 camRot0  = cam2volRotGPU.s012;
    const float3 camRot1  = cam2volRotGPU.s456;
    const float3 camRot2  = cam2volRotGPU.s89a;
    //const float3 camTrans = cam2volRotGPU.s37b;

    const float3 volRot0  = vol2camRotGPU.s012;
    const float3 volRot1  = vol2camRotGPU.s456;
    const float3 volRot2  = vol2camRotGPU.s89a;
    const float3 volTrans = vol2camRotGPU.s37b;

    float3 planed = (float3)(((float2)(x, y) - cxy)*fixy, 1.f);
    planed = (float3)(dot(planed, camRot0),
                      dot(planed, camRot1),
                      dot(planed, camRot2));

    float3 orig = (float3) (cam2volTransGPU.s0, cam2volTransGPU.s1, cam2volTransGPU.s2);
    float3 dir = fast_normalize(planed);

    float tmin = 0;
    float tmax = truncateThreshold;
    float tcurr = tmin;
    float tprev = tcurr;
    float prevTsdf = truncDist;

    int3 volStrides = volStrides4.xyz;

    while (tcurr < tmax)
    {
        float3 currRayPos = orig + tcurr * dir;

        // VolumeToVolumeUnitIdx()
        int3 currVolumeUnitIdx = (int3) (
        floor (currRayPos.x / volumeUnitSize),
        floor (currRayPos.y / volumeUnitSize),
        floor (currRayPos.z / volumeUnitSize) );
        
        int row = findRow(hash_table, currVolumeUnitIdx, list_size, bufferNums, hash_divisor);
        float currTsdf = prevTsdf;
        int currWeight = 0;
        float stepSize = 0.5 * volumeUnitSize;
        int3 volUnitLocalIdx;

        if (row >= 0)
        {
            //TsdfVoxel currVoxel
            // VolumeUnitIdxToVolume()
            float3 currVolUnitPos = (float3) 
            (( (float) (currVolumeUnitIdx.s0) * volumeUnitSize), 
             ( (float) (currVolumeUnitIdx.s1) * volumeUnitSize), 
             ( (float) (currVolumeUnitIdx.s2) * volumeUnitSize) );
            
            // VolumeToVoxelCoord()
            float3 pos = currRayPos - currVolUnitPos;
            volUnitLocalIdx = (int3)
            (( floor ( (float) (pos.s0) * voxelSizeInv) ), 
             ( floor ( (float) (pos.s1) * voxelSizeInv) ), 
             ( floor ( (float) (pos.s2) * voxelSizeInv) ) );

            struct TsdfVoxel currVoxel  = at(volUnitLocalIdx, row, volumeUnitResolution, volStrides, allVolumePtr, table_offset);

            currTsdf = tsdfToFloat(currVoxel.tsdf);
            currWeight = currVoxel.weight;
            stepSize = tstep;
        }

        if (prevTsdf > 0.f && currTsdf <= 0.f && currWeight > 0)
        {
            float tInterp = (tcurr * prevTsdf - tprev * currTsdf) / (prevTsdf - currTsdf);
            if ( !isnan(tInterp) && !isinf(tInterp) )
            {
                int3 volResolution = (int3) (volResolution4.s0, volResolution4.s1, volResolution4.s2);

                float3 pv = orig + tInterp * dir;
                float3 nv = getNormalVoxel( pv, allVolumePtr, volResolution,
                                            voxelSizeInv, hash_table,
                                            list_size, bufferNums, hash_divisor,
                                            volStrides, table_offset);

                if(!any(isnan(nv)))
                {
                    //convert pv and nv to camera space
                    normal = (float3)(dot(nv, volRot0),
                                      dot(nv, volRot1),
                                      dot(nv, volRot2));
                    // interpolation optimized a little
                    point = (float3)(dot(pv, volRot0),
                                     dot(pv, volRot1),
                                     dot(pv, volRot2)) + volTrans;
                }
            }
            break;
        }
        prevTsdf = currTsdf;
        tprev = tcurr;
        tcurr += stepSize;
    }

    __global float* pts = (__global float*)(pointsptr  +  points_offset + y*points_step   + x*sizeof(ptype));
    __global float* nrm = (__global float*)(normalsptr + normals_offset + y*normals_step  + x*sizeof(ptype));
    vstore4((float4)(point,  0), 0, pts);
    vstore4((float4)(normal, 0), 0, nrm);
}