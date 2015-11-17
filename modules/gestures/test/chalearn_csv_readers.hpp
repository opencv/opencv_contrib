#ifndef __OPENCV_CHALEARN_CSV_READERS_HPP__
#define __OPENCV_CHALEARN_CSV_READERS_HPP__

#ifdef __cplusplus

#include <string>
#include <fstream>
#include <vector>

#include <opencv2/core.hpp>

namespace cvtest
{
    class MocapCSVStreamer
    {
        public:
            MocapCSVStreamer(std::string filePath);

            bool isOpened();

            bool read(cv::OutputArray skelFrame);

        private:
            std::ifstream mStream;
    };

    class MetaDataCSVReader
    {
        public:
            MetaDataCSVReader(std::string filePath);

            inline bool isValid() const
            {
                return mValid;
            }

            inline int getFrameCount() const
            {
                return mFrameCount;
            }

            inline int getFPS() const
            {
                return mFPS;
            }

            inline int getMaxDepth() const
            {
                return mMaxDepth;
            }

        private:
            bool mValid;

            int mFrameCount;
            int mFPS;
            int mMaxDepth;
    };

    class LabelsCSVReader
    {
        public:
            LabelsCSVReader(std::string filePath, int frameCount);

            inline bool isValid() const
            {
                return mValid;
            }

            inline int getLabel(int frame) const
            {
                return mLabels[frame];
            }

        private:
            std::vector<int> mLabels;
            bool mValid;
    };
}

#endif // __cplusplus
#endif // __OPENCV_CHALEARN_CSV_READER_HPP__
