// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <iostream>
#include <unordered_map>

#include "opencv2/core/types.hpp"

#if defined(HAVE_EIGEN)
#include <Eigen/Core>
#include "opencv2/core/eigen.hpp"
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#endif

namespace cv
{
namespace kinfu
{
/*!
 * \class BlockSparseMat
 * Naive implementation of Sparse Block Matrix
 */
template<typename _Tp, int blockM, int blockN>
struct BlockSparseMat
{

    struct Point2iHash
    {
        size_t operator()(const cv::Point2i& point) const noexcept
        {
            size_t seed                     = 0;
            constexpr uint32_t GOLDEN_RATIO = 0x9e3779b9;
            seed ^= std::hash<int>()(point.x) + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
            seed ^= std::hash<int>()(point.y) + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
    typedef std::unordered_map<Point2i, Matx<_Tp, blockM, blockN>, Point2iHash> IDtoBlockValueMap;
    static constexpr int blockSize = blockM * blockN;
    BlockSparseMat(int _nBlocks) : nBlocks(_nBlocks), ijValue() {}

    Matx66f& refBlock(int i, int j)
    {
        Point2i p(i, j);
        auto it = ijValue.find(p);
        if (it == ijValue.end())
        {
            it = ijValue.insert({ p, Matx<_Tp, blockM, blockN>() }).first;
        }
        return it->second;
    }

    float& refElem(int i, int j)
    {
        Point2i ib(i / blockSize, j / blockSize), iv(i % blockSize, j % blockSize);
        return refBlock(ib.x, ib.y)(iv.x, iv.y);
    }

#if defined (HAVE_EIGEN)
    Eigen::SparseMatrix<_Tp> toEigen() const
    {
        std::vector<Eigen::Triplet<double>> tripletList;
        tripletList.reserve(ijValue.size() * blockSize * blockSize);

        for (auto ijv : ijValue)
        {
            int xb = ijv.first.x, yb = ijv.first.y;
            Matx66f vblock = ijv.second;
            for (int i = 0; i < blockSize; i++)
            {
                for (int j = 0; j < blockSize; j++)
                {
                    float val = vblock(i, j);
                    if (abs(val) >= NON_ZERO_VAL_THRESHOLD)
                    {
                        tripletList.push_back(Eigen::Triplet<double>(blockSize * xb + i, blockSize * yb + j, val));
                    }
                }
            }
        }
        Eigen::SparseMatrix<_Tp> EigenMat(blockSize * nBlocks, blockSize * nBlocks);
        EigenMat.setFromTriplets(tripletList.begin(), tripletList.end());
        EigenMat.makeCompressed();

        return EigenMat;
    }
#endif
    size_t nonZeroBlocks() const { return ijValue.size(); }

    static constexpr float NON_ZERO_VAL_THRESHOLD = 0.0001f;
    int nBlocks;
    IDtoBlockValueMap ijValue;
};

//! Function to solve a sparse linear system of equations HX = B
//! Requires Eigen
static bool sparseSolve(const BlockSparseMat<float, 6, 6>& H, const Mat& B, Mat& X)
{
    bool result = false;

#if defined(HAVE_EIGEN)
    std::cout << "starting eigen-insertion..." << std::endl;

    Eigen::SparseMatrix<float> bigA = H.toEigen();
    Eigen::VectorXf bigB;
    cv2eigen(B, bigB);

    //!TODO: Check if this is required
    Eigen::SparseMatrix<float> bigATranspose = bigA.transpose();
    if (!bigA.isApprox(bigATranspose))
    {
        CV_Error(Error::StsBadArg, "Sparse Matrix is not symmetric");
        return result;
    }


    // TODO: try this, LLT and Cholesky
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;

    std::cout << "starting eigen-compute..." << std::endl;
    solver.compute(bigA);

    if (solver.info() != Eigen::Success)
    {
        std::cout << "failed to eigen-decompose" << std::endl;
        result = false;
    }
    else
    {
        std::cout << "starting eigen-solve..." << std::endl;

        Eigen::VectorXf solutionX = solver.solve(bigB);
        if (solver.info() != Eigen::Success)
        {
            std::cout << "failed to eigen-solve" << std::endl;
            result = false;
        }
        else
        {
            eigen2cv(solutionX, X);
            result = true;
        }
    }
#else
    std::cout << "no eigen library" << std::endl;
    CV_Error(Error::StsNotImplemented, "Eigen library required for matrix solve, dense solver is not implemented");
#endif
    return result;
}

}  // namespace kinfu
}  // namespace cv
