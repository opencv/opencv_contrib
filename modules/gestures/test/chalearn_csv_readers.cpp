#include "chalearn_csv_readers.hpp"
#include <opencv2/gestures/skeleton_frame.hpp>

#include <sstream>
#include <cstdlib>

namespace cvtest
{
    MocapCSVStreamer::MocapCSVStreamer(std::string filePath):
        mStream(filePath.c_str())
    {
    }

    bool MocapCSVStreamer::isOpened()
    {
        return mStream.is_open();
    }

    bool MocapCSVStreamer::read(cv::OutputArray skelFrame)
    {
        if(!mStream.is_open())
        {
            return false;
        }

        std::string row;
        if(!std::getline(mStream, row))
        {
            return false;
        }
        std::istringstream rowStream(row);

        std::string cell;

        skelFrame.create(cv::Size(cv::gestures::SkeletonFrame::JOINTS_COUNT, 5), CV_32F);
        cv::Mat data = skelFrame.getMat();
        int joint = 0;
        int coord = 0;

        for(int j = 0; j < 17; ++j)
        {
            for(int i = 0; i < 9; ++i)
            {
                std::getline(rowStream, cell, ',');

                //  0 HipCenter
                //  1 Spine
                //  2 ShoulderCenter
                //  3 Head
                //  4 ShoulderLeft
                //  5 ElbowLeft
                //  6 WristLeft
                //  7 HandLeft
                //  8 ShoulderRight
                //  9 ElbowRight
                // 10 WristRight
                // 11 HandRight
                // 12 HipLeft
                // 13 KneeLeft
                // 14 AnkleLeft
                // 15 FootLeft
                // 16 HipRight
                // 17 KneeRight
                // 18 AnkleRight
                // 19 FootRight

                // Skip : Spine, Wrists, Knee-, Ankle- and Foot-Left
                if(j == 1 || j == 6 || j == 10 || j == 13 || j == 14 || j == 15)
                {
                    continue;
                }

                // Skip : Joint orientation
                if(i >= 3 && i < 7)
                {
                    continue;
                }

                data.at<float>(coord++, joint) = atof(cell.c_str());

                if(coord >= data.rows)
                {
                    coord = 0;
                    ++joint;
                }
            }
        }

        return true;
    }



    MetaDataCSVReader::MetaDataCSVReader(std::string filePath):
        mValid(false)
    {
        std::ifstream filestream(filePath.c_str());

        if(filestream.is_open())
        {
            std::string elem;
            std::getline(filestream, elem, ',');
            mFrameCount = atoi(elem.c_str());

            std::getline(filestream, elem, ',');
            mFPS = atoi(elem.c_str());

            std::getline(filestream, elem, ',');
            mMaxDepth = atoi(elem.c_str());

            mValid = true;
        }
    }



    LabelsCSVReader::LabelsCSVReader(std::string filePath, int frameCount):
        mLabels(frameCount, 0),
        mValid(false)
    {
        std::ifstream filestream(filePath.c_str());

        if(filestream.is_open())
        {
            std::string line;
            std::string elem;
            int label;
            int begin;
            int end;

            int last = 0;

            while(std::getline(filestream, line))
            {
                std::istringstream linestream(line);
                std::getline(linestream, elem, ',');
                label = atoi(elem.c_str());
                std::getline(linestream, elem, ',');
                begin = atoi(elem.c_str());
                std::getline(linestream, elem, ',');
                end = atoi(elem.c_str());

                for(int i = last; i < begin-1; ++i)
                {
                    mLabels[i] = 0;
                }

                for(int i = begin-1; i < end-1; ++i)
                {
                    mLabels[i] = label;
                }

                last = end;
            }

            mValid = true;
        }
    }
}
