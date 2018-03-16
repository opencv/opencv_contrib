//TODO: add license

#include "precomp.hpp"

using namespace cv;
using namespace cv::kinfu;
using namespace std;

ICP::ICP(const Intr _intrinsics, const std::vector<int>& _iterations, float _angleThreshold, float _distanceThreshold) :
    iterations(_iterations), angleThreshold(_angleThreshold), distanceThreshold(_distanceThreshold),
    intrinsics(_intrinsics)
{ }


bool ICP::estimateTransform(cv::Affine3f& transform,
                            const std::vector<Points>& oldPoints, const std::vector<Normals>& oldNormals,
                            const std::vector<Points>& newPoints, const std::vector<Normals>& newNormals)
{
    transform = Affine3f::Identity();
    for(int level = iterations.size() - 1; level >= 0; level--)
    {
        Points  oldPts = oldPoints [level], newPts = newPoints [level];
        Normals oldNrm = oldNormals[level], newNrm = newNormals[level];

        for(int iter = 0; iter < iterations[level]; iter++)
        {
            Matx66f A;
            Vec6f b;

            getAb(oldPts, oldNrm, newPts, newNrm, transform, level, A, b);

            double det = cv::determinant(A);

            if (abs (det) < 1e-15 || cvIsNaN(det))
                return false;

            Vec6f x;
            //TODO: other methods of solving?
            solve(A, b, x, DECOMP_SVD);
            Affine3f tinc(Vec3f(x.val), Vec3f(x.val+3));
            transform = tinc * transform;
        }
    }

    return true;
}

typedef Points::value_type p3type;
const float qnan = std::numeric_limits<kftype>::quiet_NaN();
const p3type qnan3(qnan, qnan, qnan);

inline p3type bilinear3(const Points& points, Point2f pt)
{
    if(pt.x < 0 || pt.x >= points.cols-1 ||
       pt.y < 0 || pt.y >= points.rows-1)
        return qnan3;

    int xi = cvFloor(pt.x), yi = cvFloor(pt.y);
    float tx = pt.x - xi, ty = pt.y - yi;

    p3type v00 = points(Point(xi+0, yi+0));
    p3type v01 = points(Point(xi+1, yi+0));
    p3type v10 = points(Point(xi+0, yi+1));
    p3type v11 = points(Point(xi+1, yi+1));

    bool b00 = !isNaN(v00);
    bool b01 = !isNaN(v01);
    bool b10 = !isNaN(v10);
    bool b11 = !isNaN(v11);

    //fix missing data
    int nz = b00 + b01 + b10 + b11;
    if(nz == 0)
    {
        return qnan3;
    }
    if(nz == 1)
    {
        if(b00) return v00;
        if(b01) return v01;
        if(b10) return v10;
        if(b11) return v11;
    }
    else if(nz == 2)
    {
        if(b00 && b10) v01 = v00, v11 = v10;
        if(b01 && b11) v00 = v01, v10 = v11;
        if(b00 && b01) v10 = v00, v11 = v01;
        if(b10 && b11) v00 = v10, v01 = v11;
        if(b00 && b11) v01 = v10 = (v00 + v11)*0.5f;
        if(b01 && b10) v00 = v11 = (v01 + v10)*0.5f;
    }
    else if(nz == 3)
    {
        if(!b00) v00 = v10 + v01 - v11;
        if(!b01) v01 = v00 + v11 - v10;
        if(!b10) v10 = v00 + v11 - v01;
        if(!b11) v11 = v01 + v10 - v00;
    }
    return v00*(1.f-tx)*(1.f-ty) + v01*tx*(1.f-ty) + v10*(1.f-tx)*ty + v11*tx*ty;
}

void ICP::getAb(const Points oldPts, const Normals oldNrm, const Points newPts, const Normals newNrm,
                Affine3f pose, int level, Matx66f &A, Vec6f &b)
{
    CV_Assert(oldPts.size() == oldNrm.size());
    CV_Assert(newPts.size() == newNrm.size());

    //TODO: optimize it to use only 27 elems
    typedef Matx<float, 6, 7> ABtype;
    ABtype sumAB = ABtype::zeros();

    Intr::Projector proj = intrinsics.scale(level).makeProjector();

    float sqDistanceThresh = distanceThreshold*distanceThreshold;
    float minCos = cos(angleThreshold);

    for(int y = 0; y < newPts.rows; y++)
    {
        const p3type* newPtsRow = newPts[y];
        const p3type* newNrmRow = newNrm[y];

        for(int x = 0; x < newPts.cols; x++)
        {
            p3type newP = newPtsRow[x];
            p3type newN = newNrmRow[x];

            p3type oldP(qnan3), oldN(qnan3);

            if(!(isNaN(newP) || isNaN(newN)))
            {
                //transform to old coord system
                newP = pose * newP;
                newN = pose.rotation() * newN;

                //find correspondence
                Point2f oldCoords = proj(newP);
                oldP = bilinear3(oldPts, oldCoords);
                oldN = bilinear3(oldNrm, oldCoords);
            }
            else
            {
                continue;
            }

            if(!(isNaN(oldP) || isNaN(oldN)))
            {
                //filter by distance
                if((newP - oldP).dot(newP - oldP) > sqDistanceThresh)
                {
                    continue;
                }

                //filter by angle
                if(abs(newN.dot(oldN)) < minCos)
                {
                    continue;
                }

                // build point-wise vector ab = [ A | b ]
                Point3f VxN = newP.cross(oldN);
                Vec<float, 7> ab(VxN.x, VxN.y, VxN.z, oldN.x, oldN.y, oldN.z, oldN.dot(oldP - newP));

                // build point-wise upper-triangle matrix [ab^T * ab] w/o last row
                // which is [A^T*A | A^T*b]
                //TODO: optimize it to use only 27 elems
                ABtype aab = ABtype::zeros();
                for(int i = 0; i < 6; i++)
                {
                    for(int j = i; j < 7; j++)
                    {
                        aab(i, j) = ab(i)*ab(j);
                    }
                }
                //TODO: optimize it to use only 27 elems
                sumAB += aab;
            }
        }
    }

    // splitting AB matrix to A and b
    for(int i = 0; i < 6; i++)
    {
        // augment lower triangle of A by symmetry
        for(int j = i; j < 6; j++)
        {
            A(i, j) = A(j, i) = sumAB(i, j);
        }

        b(i) = sumAB(i, 6);
    }
}
