//TODO: add license

#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#include "opencv2/imgproc.hpp"
#include "opencv2/kinect_fusion/kinfu.hpp"

//DEBUG
//TODO: remove it
#include <iostream>

typedef float depthType;

const float qnan = std::numeric_limits<float>::quiet_NaN();
const cv::Vec3f nan3(qnan, qnan, qnan);

inline bool isNaN(cv::Point3f p)
{
    return (cvIsNaN(p.x) || cvIsNaN(p.y) || cvIsNaN(p.z));
}

//TODO: remove it
//debugging code

struct ScopeTime
{
    static int nested;
    ScopeTime(std::string name_, bool _enablePrint = true) : name(name_), enablePrint(_enablePrint)
    {
        start = (double)cv::getTickCount();
        nested++;
    }

    ~ScopeTime()
    {
        double time_ms =  ((double)cv::getTickCount() - start)*1000.0/cv::getTickFrequency();
        if(enablePrint)
        {
            std::string spaces(nested, '-');
            std::cout << spaces << "Time(" << name << ") = " << time_ms << " ms" << std::endl;
        }
        nested--;
    }

    const std::string name;
    const bool enablePrint;
    double start;
};

/*
struct SampledScopeTime
{
public:
    enum { ENABLE_PRINT = true, EACH = 33 };
    SampledScopeTime(double& time_ms) : time_ms_(time_ms)
    {
        start = (double)cv::getTickCount();
    }

    ~SampledScopeTime()
    {
        static int i_ = 0;
        time_ms_ += getTime ();
        if (i_ % EACH == 0 && i_)
        {
            if(ENABLE_PRINT)
            {
                std::cout << "Average frame time = " << time_ms_ / EACH << "ms ( " << 1000.f * EACH / time_ms_ << "fps )" << std::endl;
            }
            time_ms_ = 0.0;
        }
        ++i_;
    }

private:
    double getTime()
    {
        return ((double)cv::getTickCount() - start)*1000.0/cv::getTickFrequency();
    }

    SampledScopeTime(const SampledScopeTime&);
    SampledScopeTime& operator=(const SampledScopeTime&);

    double& time_ms_;
    double start;
};
*/

#endif
