#pragma once

#include <opencv2/dynamicfusion/types.hpp>

namespace cv
{
    namespace kfusion
    {
        namespace cuda
        {
            class ProjectiveICP
            {
            public:
                enum { MAX_PYRAMID_LEVELS = 4 };

                typedef std::vector<Depth> DepthPyr;
                typedef std::vector<Cloud> PointsPyr;
                typedef std::vector<Normals> NormalsPyr;

                ProjectiveICP();
                virtual ~ProjectiveICP();

                float getDistThreshold() const;
                void setDistThreshold(float distance);

                float getAngleThreshold() const;
                void setAngleThreshold(float angle);

                void setIterationsNum(const std::vector<int>& iters);
                int getUsedLevelsNum() const;

                virtual bool estimateTransform(Affine3f& affine, const Intr& intr, const Frame& curr, const Frame& prev);

                /** The function takes masked depth, i.e. it assumes for performance reasons that
                  * "if depth(y,x) is not zero, then normals(y,x) surely is not qnan" */
                virtual bool estimateTransform(Affine3f& affine, const Intr& intr, const DepthPyr& dcurr, const NormalsPyr ncurr, const DepthPyr dprev, const NormalsPyr nprev);
                virtual bool estimateTransform(Affine3f& affine, const Intr& intr, const PointsPyr& vcurr, const NormalsPyr ncurr, const PointsPyr vprev, const NormalsPyr nprev);

                //static Vec3f rodrigues2(const Mat3f& matrix);
            private:
                std::vector<int> iters_;
                float angle_thres_;
                float dist_thres_;
                DeviceArray2D<float> buffer_;

                struct StreamHelper;
                cv::Ptr<StreamHelper> shelp_;
            };
        }
    }
}
