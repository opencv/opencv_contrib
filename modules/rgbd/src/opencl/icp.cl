// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

//TODO: replace by real reduce code

#define UTSIZE 27

typedef float4 ptype;

__kernel void getAbNoReduce(__global const char * oldPointsptr,
                            int oldPoints_step, int oldPoints_offset,
                            int oldPoints_rows, int oldPoints_cols,
                            __global const char * oldNormalsptr,
                            int oldNormals_step, int oldNormals_offset,
                            int oldNormals_rows, int oldNormals_cols,
                            __global const char * newPointsptr,
                            int newPoints_step, int newPoints_offset,
                            int newPoints_rows, int newPoints_cols,
                            __global const char * newNormalsptr,
                            int newNormals_step, int newNormals_offset,
                            int newNormals_rows, int newNormals_cols,
                            __global const float * poseptr,
                            const float fx, const float fy,
                            const float cx, const float cy,
                            const float sqDistanceThresh,
                            const float minCos,
                            //TODO: remove it later
                            __global char * reduceableptr,
                            int reduceable_step, int reduceable_offset,
                            int reduceable_rows, int reduceable_cols
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= newPoints_cols || y >= newPoints_rows)
        return;

    __global const float* pm = poseptr;
    const float3 poseRot0  = vload4(0, pm).xyz;
    const float3 poseRot1  = vload4(1, pm).xyz;
    const float3 poseRot2  = vload4(2, pm).xyz;
    const float3 poseTrans = (float3)(pm[3], pm[7], pm[11]);

    const float2 fxy = (float2)(fx, fy);
    const float2 cxy = (float2)(cx, cy);
    const float2 oldEdge = (float2)(oldPoints_cols - 1, oldPoints_rows - 1);

    __global const ptype* newPtsRow = (__global const ptype*)(newPointsptr +
                                                              newPoints_offset +
                                                              y*newPoints_step);

    __global const ptype* newNrmRow = (__global const ptype*)(newNormalsptr +
                                                              newNormals_offset +
                                                              y*newNormals_step);

    float3 newP = newPtsRow[x].xyz;
    float3 newN = newNrmRow[x].xyz;

    if(any(isnan(newP)) || any(isnan(newN)))
        return;

    //transform to old coord system
    newP = (float3)(dot(newP, poseRot0),
                    dot(newP, poseRot1),
                    dot(newP, poseRot2)) + poseTrans;
    newN = (float3)(dot(newN, poseRot0),
                    dot(newN, poseRot1),
                    dot(newN, poseRot2));

    //find correspondence by projecting the point
    float2 oldCoords = mad((newP.xy/newP.z), fxy, cxy);

    if(!(all(oldCoords >= 0) && all(oldCoords < oldEdge)))
        return;

    // bilinearly interpolate oldPts and oldNrm under oldCoords point
    float3 oldP, oldN;
    float2 ip = floor(oldCoords);
    float2 t = oldCoords - ip;
    int xi = ip.x, yi = ip.y;

    __global const ptype* prow0 = (__global const ptype*)(oldPointsptr +
                                                          oldPoints_offset +
                                                          (yi+0)*oldPoints_step);
    __global const ptype* prow1 = (__global const ptype*)(oldPointsptr +
                                                          oldPoints_offset +
                                                          (yi+1)*oldPoints_step);
    float3 p00 = prow0[xi+0].xyz;
    float3 p01 = prow0[xi+1].xyz;
    float3 p10 = prow1[xi+0].xyz;
    float3 p11 = prow1[xi+1].xyz;

    // NaN check is done later

    __global const ptype* nrow0 = (__global const ptype*)(oldNormalsptr +
                                                          oldNormals_offset +
                                                          (yi+0)*oldNormals_step);
    __global const ptype* nrow1 = (__global const ptype*)(oldNormalsptr +
                                                          oldNormals_offset +
                                                          (yi+1)*oldNormals_step);

    float3 n00 = nrow0[xi+0].xyz;
    float3 n01 = nrow0[xi+1].xyz;
    float3 n10 = nrow1[xi+0].xyz;
    float3 n11 = nrow1[xi+1].xyz;

    // NaN check is done later

    float3 p0 = mix(p00, p01, t.x);
    float3 p1 = mix(p10, p11, t.x);
    oldP = mix(p0, p1, t.y);

    float3 n0 = mix(n00, n01, t.x);
    float3 n1 = mix(n10, n11, t.x);
    oldN = mix(n0, n1, t.y);

    if(any(isnan(oldP)) || any(isnan(oldN)))
        return;

    //filter by distance
    float3 diff = newP - oldP;
    if(dot(diff, diff) > sqDistanceThresh)
        return;

    //filter by angle
    if(fabs(dot(newN, oldN)) < minCos)
        return;

    //TODO: when reduce do not forget to init this with zeros
    float upperTriangle[UTSIZE];

    // build point-wise vector ab = [ A | b ]

    float3 VxN = cross(newP, oldN);
    float ab[7] = {VxN.x, VxN.y, VxN.z, oldN.x, oldN.y, oldN.z, -dot(oldN, diff)};

    // build point-wise upper-triangle matrix [ab^T * ab] w/o last row
    // which is [A^T*A | A^T*b]
    // and gather sum
    int pos = 0;
    for(int i = 0; i < 6; i++)
    {
        for(int j = i; j < 7; j++)
        {
            upperTriangle[pos++] = ab[i]*ab[j];
        }
    }

    //TODO: remove it and make real reduce

    __global float* rd = (__global float*)(reduceableptr + reduceable_offset +
                                           y*reduceable_step +
                                           x*UTSIZE*sizeof(float));
    for(int i = 0; i < UTSIZE; i++)
    {
        rd[i] = upperTriangle[i];
    }
}
