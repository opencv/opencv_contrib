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


__kernel void raycast(
                    __global struct Volume_NODE * hash_table,
                    const int list_size, 
                    const int bufferNums, 
                    const int hash_divisor,
                    //__global const int4 * totalVolUnits,
                    __global float4 * points,
                        int points_step, int points_offset,
                        int points_rows, int points_cols,
                    __global float4 * normals,
                        int normals_step, int normals_offset,
                        int normals_rows, int normals_cols,
                    const int2 frameSize,
                    __global const float * allVol2cam,
                        int val2cam_step, int val2cam_offset,
                        int val2cam_rows, int val2cam_cols,
                    __global const float * allCam2vol,
                        int cam2vol_step, int cam2vol_offset,
                        int cam2vol_rows, int cam2vol_cols,
                    __global const float * vol2camptr,
                    __global const float * cam2volptr,
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
                    float voxelSizeInv
                    //, int test
                    )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    //printf(" %d \n", test);
    //printf("raycast_GPU \n");
    //float4 point = points[points_offset + (i) * points_step];
    //printf("point (%d) = [%f, %f, %f, %f] \n", i, point[0], point[1], point[2], point[3]);
    
    /*
    printf(" cam2volTransGPU [%f, %f, %f, %f] \n cam2volRotGPU | %f, %f, %f, %f | %f, %f, %f, %f | %f, %f, %f, %f | %f, %f, %f, %f | \n vol2camRotGPU | %f, %f, %f, %f | %f, %f, %f, %f | %f, %f, %f, %f | %f, %f, %f, %f | \n ", 
    cam2volTransGPU[0], cam2volTransGPU[1], cam2volTransGPU[2], cam2volTransGPU[3],

    cam2volRotGPU[0],cam2volRotGPU[1],cam2volRotGPU[2],cam2volRotGPU[3],cam2volRotGPU[4],cam2volRotGPU[5],cam2volRotGPU[6],cam2volRotGPU[7],
    cam2volRotGPU[8],cam2volRotGPU[9],cam2volRotGPU[10],cam2volRotGPU[11],cam2volRotGPU[12],cam2volRotGPU[13],cam2volRotGPU[14],cam2volRotGPU[15], 
    
    vol2camRotGPU[0],vol2camRotGPU[1],vol2camRotGPU[2],vol2camRotGPU[3],vol2camRotGPU[4],vol2camRotGPU[5],vol2camRotGPU[6],vol2camRotGPU[7],
    vol2camRotGPU[8],vol2camRotGPU[9],vol2camRotGPU[10],vol2camRotGPU[11],vol2camRotGPU[12],vol2camRotGPU[13],vol2camRotGPU[14],vol2camRotGPU[15] 
    );
    */


    if(x >= frameSize.x || y >= frameSize.y)
        return;



    __global const float* cm = cam2volptr;
    const float3 camRot0  = vload4(0, cm).xyz;
    const float3 camRot1  = vload4(1, cm).xyz;
    const float3 camRot2  = vload4(2, cm).xyz;
    const float3 camTrans = (float3)(cm[3], cm[7], cm[11]);

    __global const float* vm = vol2camptr;
    const float3 volRot0  = vload4(0, vm).xyz;
    const float3 volRot1  = vload4(1, vm).xyz;
    const float3 volRot2  = vload4(2, vm).xyz;
    const float3 volTrans = (float3)(vm[3], vm[7], vm[11]);

    /*
    printf("camRot0 [%f, %f, %f] \ncamRot1 [%f, %f, %f] \ncamRot2 [%f, %f, %f] \ncamTrans [%f, %f, %f] \n", 
    camRot0[0], camRot0[1], camRot0[2],
    camRot1[0], camRot1[1], camRot1[2],
    camRot2[0], camRot2[1], camRot2[2],
    camTrans[0], camTrans[1], camTrans[2]
    );
    */


    float3 planed = (float3)(((float2)(x, y) - cxy)*fixy, 1.f);
    planed = (float3)(dot(planed, camRot0),
                      dot(planed, camRot1),
                      dot(planed, camRot2));

    float3 orig = (float3) (1, 1, 1);
    float3 dir = fast_normalize(planed);

    float tmin = 0;
    truncateThreshold = 4;
    float tmax = truncateThreshold;

    float tcurr = tmin;
    float tprev = tcurr;

    if (tcurr < tmax)
    {
        
        float3 currRayPos = orig + tcurr * dir;

        int3 point = (int3) (
        (int) currRayPos.x * voxelSizeInv,
        (int) currRayPos.y * voxelSizeInv,
        (int) currRayPos.z * voxelSizeInv
        );

        printf("lol [%d, %d] tcurr=%d tmax=%d [%d, %d, %d] \n", x, y, tcurr, tmax, point.x, point.y, point.z);
        
        tprev = tcurr;
        tcurr += tstep;
    }

    



    // coordinate-independent constants
    /*
    __global const float* cm = cam2volptr;
    const float3 camRot0  = vload4(0, cm).xyz;
    const float3 camRot1  = vload4(1, cm).xyz;
    const float3 camRot2  = vload4(2, cm).xyz;
    const float3 camTrans = (float3)(cm[3], cm[7], cm[11]);

    __global const float* vm = vol2camptr;
    const float3 volRot0  = vload4(0, vm).xyz;
    const float3 volRot1  = vload4(1, vm).xyz;
    const float3 volRot2  = vload4(2, vm).xyz;
    const float3 volTrans = (float3)(vm[3], vm[7], vm[11]);

    const float3 boxDown = boxDown4.xyz;
    const float3 boxUp   = boxUp4.xyz;
    const int3   volDims = volDims4.xyz;

    const int3 volResolution = volResolution4.xyz;

    const float invVoxelSize = native_recip(voxelSize);

    // kernel itself

    float3 point  = nan((uint)0);
    float3 normal = nan((uint)0);

    float3 orig = camTrans;

    // get direction through pixel in volume space:
    // 1. reproject (x, y) on projecting plane where z = 1.f
    float3 planed = (float3)(((float2)(x, y) - cxy)*fixy, 1.f);

    // 2. rotate to volume space
    planed = (float3)(dot(planed, camRot0),
                      dot(planed, camRot1),
                      dot(planed, camRot2));

    // 3. normalize
    float3 dir = fast_normalize(planed);

    // compute intersection of ray with all six bbox planes
    float3 rayinv = native_recip(dir);
    float3 tbottom = rayinv*(boxDown - orig); //incorrect value
    float3 ttop    = rayinv*(boxUp   - orig); //incorrect value

    // re-order intersections to find smallest and largest on each axis
    float3 minAx = min(ttop, tbottom);
    float3 maxAx = max(ttop, tbottom);

    // near clipping plane
    const float clip = 0.f;
    float tmin = max(max(max(minAx.x, minAx.y), max(minAx.x, minAx.z)), clip);
    float tmax =     min(min(maxAx.x, maxAx.y), min(maxAx.x, maxAx.z));

    // precautions against getting coordinates out of bounds
    tmin = tmin + tstep;
    tmax = tmax - tstep;
    
    //printf("GPU [%d, %d] tmin=%f tmax=%f minAx(%f, %f, %f) maxAx(%f, %f, %f) \n", 
    //    x, y, tmin, tmax, minAx.x, minAx.y, minAx.z, maxAx.x, maxAx.y, maxAx.z);
    
    //printf("GPU [%d, %d] rayinv(%f, %f, %f) tbottom(%f, %f, %f) ttop(%f, %f, %f) \n",
    //    x, y, rayinv.x, rayinv.y, rayinv.z, tbottom.x, tbottom.y, tbottom.z, ttop.x, ttop.y, ttop.z);

    printf("GPU [%d, %d] orig(%f, %f, %f) boxDown(%f, %f, %f) boxUp(%f, %f, %f) \n",
        x, y, orig.x, orig.y, orig.z, boxDown.x, boxDown.y, boxDown.z, boxUp.x, boxUp.y, boxUp.z);
    

    if(tmin < tmax)
    {
        // interpolation optimized a little
        orig *= invVoxelSize;
        dir  *= invVoxelSize;

        float3 rayStep = dir*tstep;
        float3 next = (orig + dir*tmin);

        float3 currRayPos = orig + tmin * dir;

        printf("[%f, %f, %f] \n",currRayPos[0], currRayPos[1], currRayPos[2]);

    }
    */





}