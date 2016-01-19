/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


License Agreement
For Open Source Computer Vision Library
(3-clause BSD License)

Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015, OpenCV Foundation, all rights reserved.
Copyright (C) 2015, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

* Neither the names of the copyright holders nor the names of the contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/
#include "opencv2/face/predict_collector.hpp"
#include "opencv2/core/cvstd.hpp"
#include <iterator>     // std::back_inserter
namespace cv {
namespace face {

CV_WRAP bool TopNPredictCollector::emit(const int label, const double dist, const int state)
{
    ((void)state);
    std::pair<int, double> p = std::make_pair(label, dist);
    if (_idx->size() == 0 || p.second <= _idx->front().second) {
        _idx->push_front(p);
    } else if (p.second >= _idx->back().second) {
        _idx->push_back(p);
    }
    else {
        typedef std::list<std::pair<int,double> >::iterator it_type;
        for (it_type i = _idx->begin(); i != _idx->end(); i++) {
            if (p.second <= i->second) {
                _idx->insert(i, p);
                break;
            }
        }
    }
    return true;
}

CV_WRAP bool TopNPredictCollector::filter(int * label, double * dist, const int state)
{
    ((void)state);
    if (_idx->size() < _size)return true; //not full - can insert;
    if (*dist >= _idx->back().second)return false; //too far distance
    for (std::list<std::pair<int, double> >::iterator it = _idx->begin(); it != _idx->end(); ++it) {
        if (it->first == *label) {
            if (it->second <= *dist) {
                return false; //has more close
            }
            else {
                _idx->erase(it);
                return true; //no more require pop_back
            }
        }
    }
    _idx->pop_back();
    return true;
}

CV_WRAP Ptr<std::list<std::pair<int, double> > > TopNPredictCollector::getResult()
{
    return _idx;
}

CV_WRAP std::vector<std::pair<int, double> > TopNPredictCollector::getResultVector()
{
    std::vector<std::pair<int, double> > result;
    std::copy(_idx->begin(), _idx->end(), std::back_inserter(result));
    return result;
}

CV_WRAP Ptr<TopNPredictCollector> TopNPredictCollector::create(size_t size, double threshold)
{
    return Ptr<TopNPredictCollector>(new TopNPredictCollector(size, threshold));
}

}
}