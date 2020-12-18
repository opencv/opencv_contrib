// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

//#define NAN_NUM -2147483647;

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
    int32_t isActive;
    int32_t lastVisibleIndex;
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

uint calc_hash(int4 x)
{
    uint32_t seed = 0;
    //uint GOLDEN_RATIO = 0x9e3779b9;
    uint32_t GOLDEN_RATIO = 0x9e3779b9;
    seed ^= x[0] + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
    seed ^= x[1] + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
    seed ^= x[2] + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
    return seed;
}

int findRow(__global struct Volume_NODE * hash_table, int4 indx,
               int list_size, int bufferNums, int hash_divisor)
{
    int hash = calc_hash(indx) % hash_divisor;
    
    int num = 1;
    int i = hash * num * list_size;
    int NAN_NUM = -2147483647;
    while (i != NAN_NUM)
    {
        struct Volume_NODE v = hash_table[i];
        if (v.idx[0] == indx[0] &&
            v.idx[1] == indx[1] &&
            v.idx[2] == indx[2])
            return v.row;
        if (v.idx.x == NAN_NUM)
            return -2;
        i = v.nextVolumeRow;
    }

    return -2;
}

int getIsActive(__global struct Volume_NODE * hash_table, int4 indx,
               int list_size, int bufferNums, int hash_divisor)
{
    int hash = calc_hash(indx) % hash_divisor;
    int num = 1;
    int i = hash * num * list_size;
    int NAN_NUM = -2147483647;

    while (i != NAN_NUM)
    {
        struct Volume_NODE v = hash_table[i];

        if (v.idx[0] == indx[0] &&
            v.idx[1] == indx[1] &&
            v.idx[2] == indx[2])
            return v.isActive;
        if (v.idx[0] == NAN_NUM)
            return 0;
        i = v.nextVolumeRow;
    }
    return 0;
}

void updateIsActive(__global struct Volume_NODE * hash_table, int4 indx, int isActive,
               int list_size, int bufferNums, int hash_divisor)
{
    int hash = calc_hash(indx) % hash_divisor;
    int num = 1;
    int i = hash * num * list_size;
    int NAN_NUM = -2147483647;
    while (i != NAN_NUM)
    {
        __global struct Volume_NODE * v = (hash_table + i);

        if (v->idx[0] == indx[0] &&
            v->idx[1] == indx[1] &&
            v->idx[2] == indx[2])
            v->isActive = isActive;     
        if (v->idx[0] == NAN_NUM)
            return;
        i = v->nextVolumeRow;
    }
    return;
}


void integrateVolumeUnit(
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
                        __global struct Volume_NODE * hash_table,
                        const int list_size, 
                        const int bufferNums, 
                        const int hash_divisor,
                        __global const int4 * totalVolUnits,
                        int totalVolUnits_step, int totalVolUnits_offset,
                        int totalVolUnits_rows, int totalVolUnits_cols,
                        __global struct TsdfVoxel * allVolumePtr,
                        int table_step, int table_offset,
                        int table_rows, int table_cols,
                        __global const float * pixNorms,
                        __global const float * allVol2camMatrix,
                        int val2cam_step, int val2cam_offset,
                        int val2cam_rows, int val2cam_cols,
                        const int lastVolIndex, 
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
    int k = get_global_id(2);

    int4 v = totalVolUnits[k];
    int row = findRow(hash_table, v, list_size, bufferNums, hash_divisor);
    if (row < 0 || row > lastVolIndex-1)
        return;
    
    int isActive = getIsActive(hash_table, v, list_size, bufferNums, hash_divisor);

    if (isActive == 1)
    {
        __global struct TsdfVoxel * volumeptr = (__global struct TsdfVoxel*)
                                                (allVolumePtr + table_offset + (row) * 16*16*16);
        __global const float * p_vol2camMatrix = (__global const float *)
                                                 (allVol2camMatrix + val2cam_offset + (row) * 16);
        
        const float16 vol2camMatrix = (float16) (
        p_vol2camMatrix[0], p_vol2camMatrix[1], p_vol2camMatrix[2], p_vol2camMatrix[3],
        p_vol2camMatrix[4], p_vol2camMatrix[5], p_vol2camMatrix[6], p_vol2camMatrix[7],
        p_vol2camMatrix[8], p_vol2camMatrix[9], p_vol2camMatrix[10], p_vol2camMatrix[11],
        p_vol2camMatrix[12], p_vol2camMatrix[13], p_vol2camMatrix[14], p_vol2camMatrix[15]);

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
        //updateIsActive(hash_table, v, 0, list_size, bufferNums, hash_divisor);
    }
   
}

struct TsdfVoxel _at(int3 volumeIdx, int row, 
              int volumeUnitResolution, int4 volStrides, 
              __global struct TsdfVoxel * allVolumePtr, int table_offset)

{
    //! Out of bounds
    if ((volumeIdx[0] >= volumeUnitResolution || volumeIdx[0] < 0) ||
        (volumeIdx[1] >= volumeUnitResolution || volumeIdx[1] < 0) ||
        (volumeIdx[2] >= volumeUnitResolution || volumeIdx[2] < 0))
    {
        struct TsdfVoxel dummy;
        dummy.tsdf = floatToTsdf(1.0f);
        dummy.weight = 0;
        return dummy;
    }

    __global struct TsdfVoxel * volData = (__global struct TsdfVoxel*)
                                            (allVolumePtr + table_offset + (row) * 16*16*16);    int coordBase =
        volumeIdx[0] * volStrides[0] +
        volumeIdx[1] * volStrides[1] +
        volumeIdx[2] * volStrides[2];
    return volData[coordBase];
}


struct TsdfVoxel _atVolumeUnit(int3 volumeIdx, int3 volumeUnitIdx, int row, int lastVolIndex,
              int volumeUnitResolution, int4 volStrides, 
              __global const struct TsdfVoxel * allVolumePtr, int table_offset)

{
    //! Out of bounds
    if (row < 0 || row > lastVolIndex - 1)
    {
        struct TsdfVoxel dummy;
        dummy.tsdf = floatToTsdf(1.0f);
        dummy.weight = 0;
        return dummy;
    }

    int3 volUnitLocalIdx = volumeIdx - volumeUnitIdx * volumeUnitResolution;
    __global struct TsdfVoxel * volData = (__global struct TsdfVoxel*)
                                            (allVolumePtr + table_offset + (row) * 16*16*16);    
    int coordBase =
        volUnitLocalIdx[0] * volStrides[0] +
        volUnitLocalIdx[1] * volStrides[1] +
        volUnitLocalIdx[2] * volStrides[2];
    return volData[coordBase];
}


inline float3 getNormalVoxel(float3 p, __global const struct TsdfVoxel* allVolumePtr,
                             int3 volResolution, int3 volDims, int8 neighbourCoords,
                             float voxelSizeInv, int lastVolIndex,
                             __global struct Volume_NODE * hash_table,
                             const int list_size, 
                             const int bufferNums, 
                             const int hash_divisor,
                             int4 volStrides, int table_offset)
{
    
    float3 normal = (float3) (0.0f, 0.0f, 0.0f);
    float3 ptVox = p * voxelSizeInv;
    int3 iptVox = (int3) ( floor (ptVox.x), floor (ptVox.y), floor (ptVox.z) );

    bool queried[8];
    int  iterMap[8];

    for (int i = 0; i < 8; i++)
    {
        iterMap[i] = lastVolIndex - 1;
        queried[i] = false;
    }

    int3 offsets[] = { { 1,  0,  0}, {-1,  0,  0}, { 0,  1,  0}, // 0-3
                       { 0, -1,  0}, { 0,  0,  1}, { 0,  0, -1}  // 4-7
    };
    
    const int nVals = 6;
    float vals[nVals];

    for (int i = 0; i < nVals; i++)
    {
        int3 pt = iptVox + offsets[i];

        // VoxelToVolumeUnitIdx() 
        // TODO: add assertion - if (!(vuRes & (vuRes - 1)))
        int3 volumeUnitIdx = (int3) (
            floor ( (float) pt[0] / volResolution[0]),
            floor ( (float) pt[1] / volResolution[1]),
            floor ( (float) pt[2] / volResolution[2]) );
        
        int4 volumeUnitIdx4 = (int4) (
            floor ( (float) pt[0] / volResolution[0]),
            floor ( (float) pt[1] / volResolution[1]),
            floor ( (float) pt[2] / volResolution[2]), 0 );


        int dictIdx = (volumeUnitIdx[0] & 1) 
                    + (volumeUnitIdx[1] & 1) * 2 
                    + (volumeUnitIdx[2] & 1) * 4;

        int it = iterMap[dictIdx];

        if (!queried[dictIdx])
        {
            it = findRow(hash_table, volumeUnitIdx4, list_size, bufferNums, hash_divisor);
            if (it >= 0 || it < lastVolIndex)
            {
                iterMap[dictIdx] = it;
                queried[dictIdx] = true;
            }
        }

        //_at(volUnitLocalIdx, row, volumeUnitResolution,  volStrides, allVolumePtr,  table_offset);

        struct TsdfVoxel tmp = _atVolumeUnit(pt, volumeUnitIdx, it, lastVolIndex, volResolution[0],  volStrides, allVolumePtr,  table_offset) ;
        vals[i] = tsdfToFloat( tmp.tsdf );

    }

    for (int c = 0; c < 3; c++)
    {
        normal[c] = vals[c * 2] - vals[c * 2 + 1];
    }

    float norm = 
    sqrt(normal.x*normal.x 
       + normal.y*normal.y 
       + normal.z*normal.z);
    return norm < 0.0001f ? nan((uint)0) : normal / norm;
}

typedef float4 ptype;

__kernel void raycast(
                    __global struct Volume_NODE * hash_table,
                    const int list_size, 
                    const int bufferNums, 
                    const int hash_divisor,
                    const int lastVolIndex, 
                    //__global const int4 * totalVolUnits,
                    __global float4 * points,
                        int points_step, int points_offset,
                        int points_rows, int points_cols,
                    __global float4 * normals,
                        int normals_step, int normals_offset,
                        int normals_rows, int normals_cols,
                    const int2 frameSize,
                    __global struct TsdfVoxel * allVolumePtr,
                        int table_step, int table_offset,
                        int table_rows, int table_cols,


                    //__global const float * allVol2cam,
                    //    int val2cam_step, int val2cam_offset,
                    //    int val2cam_rows, int val2cam_cols,
                    //__global const float * allCam2vol,
                    //    int cam2vol_step, int cam2vol_offset,
                    //    int cam2vol_rows, int cam2vol_cols,
                    //__global const float * vol2camptr,
                    //__global const float * cam2volptr,

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
                    int4 volStrides
                    //, int test
                    )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    // tmp posution for 1 pixel
    //x+=468; y+=29;
    //x+=168; y+=28;
    x+=300; y+=150;

    //printf("GPU voxelSizeInv=%f volumeUnitSize=%f truncDist=%f \n", voxelSizeInv, volumeUnitSize, truncDist);

    if(x >= frameSize.x || y >= frameSize.y)
        return;

    float3 point  = nan((uint)0);
    float3 normal = nan((uint)0);

    const float3 camRot0  = cam2volRotGPU.s012;
    const float3 camRot1  = cam2volRotGPU.s456;
    const float3 camRot2  = cam2volRotGPU.s89a;
    const float3 camTrans = cam2volRotGPU.s37b;

    const float3 volRot0  = vol2camRotGPU.s012;
    const float3 volRot1  = vol2camRotGPU.s456;
    const float3 volRot2  = vol2camRotGPU.s89a;
    const float3 volTrans = vol2camRotGPU.s37b;

    float3 planed = (float3)(((float2)(x, y) - cxy)*fixy, 1.f);
    planed = (float3)(dot(planed, camRot0),
                      dot(planed, camRot1),
                      dot(planed, camRot2));

    //printf("GPU camRot0=[%f, %f, %f] \n    camRot1=[%f, %f, %f] \n    camRot2=[%f, %f, %f] \n    camTrans=[%f, %f, %f] \n",
    //    camRot0[0], camRot0[1], camRot0[2], camRot1[0], camRot1[1], camRot1[2], 
    //    camRot2[0], camRot2[1], camRot2[2], camTrans[0], camTrans[1], camTrans[2] );
    
    //printf("GPU volRot0=[%f, %f, %f] \n    volRot1=[%f, %f, %f] \n    volRot2=[%f, %f, %f] \n    volTrans=[%f, %f, %f] \n",
    //    volRot0[0], volRot0[1], volRot0[2], volRot1[0], volRot1[1], volRot1[2], 
    //    volRot2[0], volRot2[1], volRot2[2], volTrans[0], volTrans[1], volTrans[2] );


    float3 orig = (float3) (cam2volTransGPU[0], cam2volTransGPU[1], cam2volTransGPU[2]);
    float3 dir = fast_normalize(planed);

    //printf("GPU [%d, %d] orig=[%f, %f, %f] dir=[%f, %f, %f] \n", x, y, orig[0], orig[1], orig[2], dir[0], dir[1], dir[2]);

    float tmin = 0;
    float tmax = truncateThreshold;
    float tcurr = tmin;
    float tprev = tcurr;
    float prevTsdf = truncDist;

    //printf("GPU [%d, %d] tmin=%f tmax=%f tcurr=%f tprev=%f prevTsdf=%f \n", x, y, tmin, tmax, tcurr, tprev, prevTsdf);

    while (tcurr < tmax)
    {
        float3 currRayPos = orig + tcurr * dir;

        // VolumeToVolumeUnitIdx()
        int3 currVolumeUnitIdx = (int3) (
        floor (currRayPos.x / volumeUnitSize),
        floor (currRayPos.y / volumeUnitSize),
        floor (currRayPos.z / volumeUnitSize) );
        
        // VolumeToVolumeUnitIdx4()
        int4 point4 = (int4) (
        floor (currRayPos.x / volumeUnitSize),
        floor (currRayPos.y / volumeUnitSize),
        floor (currRayPos.z / volumeUnitSize), 0);

        int row = findRow(hash_table, point4, list_size, bufferNums, hash_divisor);
        float currTsdf = prevTsdf;
        int currWeight = 0;
        float stepSize = 0.5 * volumeUnitSize;
        int3 volUnitLocalIdx;

        //printf("GPU [%d, %d] currRayPos=[%f, %f, %f] currVolumeUnitIdx=[%d, %d, %d] row=%d currTsdf=%f currWeight=%d stepSize=%f \n", 
        //    x, y, currRayPos[0], currRayPos[1], currRayPos[2], currVolumeUnitIdx[0], currVolumeUnitIdx[1], currVolumeUnitIdx[2], row, currTsdf, currWeight, stepSize);


        if (row >= 0 && row < lastVolIndex) {
            
            
            //TsdfVoxel currVoxel
            // VolumeUnitIdxToVolume()
            float3 currVolUnitPos = (float3) 
            (( (float) (currVolumeUnitIdx[0]) * volumeUnitSize), 
             ( (float) (currVolumeUnitIdx[1]) * volumeUnitSize), 
             ( (float) (currVolumeUnitIdx[2]) * volumeUnitSize) );
            
            // VolumeToVoxelCoord()
            float3 pos = currRayPos - currVolUnitPos;
            volUnitLocalIdx = (int3)
            (( floor ( (float) (pos[0]) * voxelSizeInv) ), 
             ( floor ( (float) (pos[1]) * voxelSizeInv) ), 
             ( floor ( (float) (pos[2]) * voxelSizeInv) ) );

            struct TsdfVoxel currVoxel  = _at(volUnitLocalIdx, row, volumeUnitResolution,  volStrides, allVolumePtr,  table_offset);

            currTsdf = tsdfToFloat(currVoxel.tsdf);
            currWeight = currVoxel.weight;
            stepSize = tstep;

            //printf("GPU [%d, %d]  row=%d currVolUnitPos=[%f, %f, %f] volUnitLocalIdx=[%d, %d, %d] currTsdf=%f currWeight=%d\n", 
            //    x, y, row, currVolUnitPos[0], currVolUnitPos[1], currVolUnitPos[2], volUnitLocalIdx[0], volUnitLocalIdx[1], volUnitLocalIdx[2], currTsdf, currWeight);
            

            //printf("GPU voxelSizeInv = %f", voxelSizeInv);
            //if (currTsdf!=1)
            //    printf("GPU [%d, %d] currTsdf=%f currWeight=%d \n", x, y, currTsdf, currWeight);
            //printf("GPU [%d, %d] currRayPos=[%f, %f, %f] idx=[%d, %d, %d] row=%d \n", x, y, currRayPos.x, currRayPos.y, currRayPos.z, point4[0], point4[1], point4[2], row);
        }

        
        if (prevTsdf > 0.f && currTsdf <= 0.f && currWeight > 0)
        {
            float tInterp = (tcurr * prevTsdf - tprev * currTsdf) / (prevTsdf - currTsdf);
            if ( !isnan(tInterp) && !isinf(tInterp) )
            {
                //if (y == 150)
                //    printf("GPU [%d, %d] tInterp=%f \n", x, y, tInterp);

                //__global struct TsdfVoxel * volumeptr = (__global struct TsdfVoxel*)
                //                                (allVolumePtr + table_offset + (row) * 16*16*16);

                int3 volResolution = (int3) (volResolution4[0], volResolution4[1], volResolution4[2]);
                int3 volDims = (int3) (volDims4[0], volDims4[1], volDims4[2]);
                
                float3 pv = orig + tInterp * dir;
                float3 nv = getNormalVoxel( pv, allVolumePtr, volResolution, volDims, neighbourCoords, 
                                            voxelSizeInv, lastVolIndex, hash_table,
                                            list_size, bufferNums, hash_divisor,
                                            volStrides, table_offset);
                //if (y == 150)
                //     printf("GPU [%d, %d] pv=[%f, %f, %f] nv=[%f, %f, %f] \n", x, y, pv[0], pv[1], pv[2], nv[0], nv[1], nv[2]);

                if(!any(isnan(nv)))
                {
                    //printf("lol \n");
                    //convert pv and nv to camera space
                    normal = (float3)(dot(nv, volRot0),
                                      dot(nv, volRot1),
                                      dot(nv, volRot2));
                    // interpolation optimized a little
                    pv *= voxelSize;
                    point = (float3)(dot(pv, volRot0),
                                     dot(pv, volRot1),
                                     dot(pv, volRot2)) + volTrans;
                }

            }
            
        }

        //printf("GPU [%d, %d] currRayPos=[%f, %f, %f] idx=[%d, %d, %d] row=%d \n", x, y, currRayPos.x, currRayPos.y, currRayPos.z, point4[0], point4[1], point4[2], row);
        //printf("[%d, %d]  currRayPos=[%f, %f, %f] voxelSizeInv=%f point=[%d, %d, %d] \n", x, y, currRayPos.x, currRayPos.y, currRayPos.z, voxelSizeInv, point.x, point.y, point.z);
        //printf("lol [%d, %d] tcurr=%f tmax=%f tstep=%f [%d, %d, %d] \n", x, y, tcurr, tmax, tstep, point.x, point.y, point.z);
        
        tprev = tcurr;
        tcurr += stepSize;
    }

    
    __global float* pts = (__global float*)(points  +  points_offset + y*points_step  + x*sizeof(ptype));
    __global float* nrm = (__global float*)(normals + normals_offset + y*normals_step + x*sizeof(ptype));
    vstore4((float4)(point,  0), 0, pts);
    vstore4((float4)(normal, 0), 0, nrm);       


}