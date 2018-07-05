// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

__kernel void integrate(__global const float * depthptr,
                        int depth_step, int depth_offset,
                        int depth_rows, int depth_cols,
/* ? volatile */        __global float * volumeptr,
                        __global const float * vol2camptr,
                        const float voxelSize,
                        const int edgeResolution,
                        const float fx, const float fy,
                        const float cx, const float cy,
                        const float dfac,
                        const float truncDist,
                        const int maxWeight)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    const float2 limits = float2(depth_cols-1, depth_rows-1);

    const float3 vol2cam0 = vload3(0, vol2camptr);
    const float3 vol2cam1 = vload3(1, vol2camptr);
    const float3 vol2cam2 = vload3(2, vol2camptr);

    const float2 fxy(fx, fy), cxy(cx, cy);

    const float truncDistInv = 1.f/truncDist;

    // optimization of camSpace transformation (vector addition instead of matmul at each z)
    float3 basePt = float3(x, y, 0)*voxelSize;
    basePt = float3(dot(vol2cam0, basePt),
                    dot(vol2cam1, basePt),
                    dot(vol2cam2, basePt));

    float3 camSpacePt = basePt;

    // zStep == vol2cam*(float3(x, y, 1)*voxelSize) - basePt;
    float3 zStep = float3(vol2cam0.z, vol2cam1.z, vol2cam2.z)*voxelSize;

    // &elem(x, y, z) = data + x*edgeRes^2 + y*edgeRes + z;
    int volYidx = (x*edgeResolution + y)*edgeResolution;

    int startZ, endZ;
    if(fabs(zStep.z) > 1e-5)
    {
        int baseZ = convert_cast<int>(-basePt.z / zStep.z);
        if(zStep.z > 0)
        {
            startZ = baseZ;
            endZ = edgeResolution;
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
            startZ = 0; endZ = volume.edgeResolution;
        }
        else
        {
            // z loop shouldn't be performed
            startZ = endZ = 0;
        }
    }

    startZ = max(0, startZ);
    endZ = min(edgeResolution, endZ);
    for(int z = startZ; z < endZ; z++)
    {
        // optimization of the following:
        //float3 volPt = float3(x, y, z)*volume.voxelSize;
        //float3 camSpacePt = vol2cam * volPt;
        camSpacePt += zStep;

        if(camSpacePt.z <= 0)
            continue;

        float3 camPixVec = camSpacePt / camSpacePt.z;
        float2 projected = fma(camPixVec.xy, fxy, cxy);

        float v;
        // bilinearly interpolate depth at projected
        if(all(projected >= 0) && all(projected < limits))
        {
            float2 ip = floor(projected);
            int xi = ip.x, yi = ip.y;

            __global const float* row0 = depthptr + (yi+0)*depth_step + depth_offset;
            __global const float* row1 = depthptr + (yi+1)*depth_step + depth_offset;

            float v00 = row0[xi+0];
            float v01 = row0[xi+1];
            float v10 = row1[xi+0];
            float v11 = row1[xi+1];
            float4 vv(v00, v01, v10, v11);

            // assume correct depth is positive
            if(all(vv > 0))
            {
                float2 t = projected - ip;
                float v0 = vv.s0 + t.x*(vv.s1 - vv.s0);
                float v1 = vv.s2 + t.x*(vv.s3 - vv.s2);

                v = v0 + t.y*(v1 - v0);

//                float2 vf = vv.xz + t.x*(vv.yw - vv.xz);
//                v = vf.s0 + t.y*(vf.s1 - vf.s0);

//                float2 vf = mix(vv.xz, vv.yw, t.x);
//                v = mix(vf.s0, vf.s1, t.y);
            }
            else
                continue;
        }
        else
            continue;

        if(v == 0)
            continue;

        float pixNorm = length(camPixVec);

        // difference between distances of point and of surface to camera
        float sdf = pixNorm*(v*dfac - camSpacePt.z);
        // possible alternative is:
        // float sdf = lenght(camSpacePt)*(v*dfac/camSpacePt.z - 1.0);

        if(sdf >= -truncDist)
        {
            float tsdf = fmin(1.0, sdf * truncDistInv);

            float2 voxel = volDataY[z];
            float value  = voxel.s0;
            int weight = as_int(voxel.s1);

            // update TSDF
            value = (value*weight + tsdf) / (weight + 1);
            weight = min(weight + 1, maxWeight);

            voxel.s0 = value;
            voxel.s1 = as_float(weight);
            volDataY[z] = voxel;
        }
    }
}
