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

#ifndef __OPENCV_PREDICT_COLLECTOR_HPP__
#define __OPENCV_PREDICT_COLLECTOR_HPP__
#include <cfloat>
#include <list>
#include <vector>
#include <map>
#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"
#undef emit //fix for qt
namespace cv {
namespace face {
//! @addtogroup face
//! @{
/** @brief Abstract base class for all strategies of prediction result handling
*/
class CV_EXPORTS_W PredictCollector {
protected:
    double _threshold;
    int _size;
    int _state;
    int _excludeLabel;
    double _distanceKoef;
    double _minthreshold;
public:
    /** @brief creates new predict collector with given threshold */
    PredictCollector(double threshold = DBL_MAX) {
        _threshold = threshold;
        _excludeLabel = 0;
        _distanceKoef = 1;
        _minthreshold = -1;
    }
    CV_WRAP virtual ~PredictCollector() {}

    /** @brief called once at start of recognition
    @param size total size of prediction evaluation that recognizer could perform
    @param state user defined send-to-back optional value to allow multi-thread, multi-session or aggregation scenarios
    */
    CV_WRAP virtual void init(const int size, const int state = 0);

    /** @brief called by recognizer prior to emit to decide if prediction require emiting
    @param label current predicted label
    @param dist current predicted distance
    @param state back send state parameter of prediction  session
    @return true if prediction is valid and required for emiting
    @note can override given label and distance to another values
    */
    CV_WRAP virtual bool defaultFilter(int* label, double* dist, const int state);

    /** @brief extension point for filter - called if base filter executed */
    CV_WRAP virtual bool filter(int* label, double* dist, const int state);

    /** @brief called with every recognition result
    @param label current prediction label
    @param dist current prediction distance (confidence)
    @param state user defined send-to-back optional value to allow multi-thread, multi-session or aggregation scenarios
    @return true if recognizer should proceed prediction , false - if recognizer should terminate prediction
    */
    CV_WRAP virtual bool emit(const int label, const double dist, const int state = 0); //not abstract while Python generation require non-abstract class

    /** @brief outer interface method to be called from recognizer
    @param label current prediction label
    @param dist current prediction distance (confidence)
    @param state user defined send-to-back optional value to allow multi-thread, multi-session or aggregation scenarios
    @note wraps filter and emit calls, not tended to be overriden
    */
    CV_WRAP virtual bool collect(int label, double dist, const int state = 0);

    /**
    @brief get size of prediction
    ### Description
    Is set by recognizer and is amount of all available predicts
    So we can use it to perform statistic collectors before prediction of whole set
    */
    CV_WRAP virtual int getSize();

    /** @brief set size of prediction */
    CV_WRAP virtual void setSize(int size);

    /**
    @brief get state of prediction
    ### Description
    State is a custom value assigned for prediction session, 0 if it's no-state session
    */
    CV_WRAP virtual int getState();

    /** @brief set state of prediction */
    CV_WRAP virtual void setState(int state);

    /**
    @brief returns currently excluded label, 0 if no set
    ### Description
    We require to exclude label if we want to test card in train set against others
    */
    CV_WRAP virtual int getExcludeLabel();

    /** @brief set exclude label of prediction */
    CV_WRAP virtual void setExcludeLabel(int excludeLabel);

    /**
    @brief returns current distance koeficient (applyed to distance in filter stage)
    ### Description
    It's required if we want to predict with distinct algorithms in one session
    so LBPH, Eigen and Fisher distance are different, but we can provide koef for them to translate to
    each other (while their distribuition for same train set is close and started from 0)
    Default 1 koef means that distance is not corrected
    */
    CV_WRAP virtual double getDistanceKoef();

    /** @brief set exclude label of prediction */
    CV_WRAP virtual void setDistanceKoef(double distanceKoef);
    /**
    @brief returns current minimal threshold
    ### Description
    It's required when we must exclude most closed predictions (for example we
    search for close but not same faces - usable for mixed set where doubles exists
    in train collection)
    */
    CV_WRAP virtual double getMinThreshold();

    /** @brief set minimal threshold for prediction */
    CV_WRAP virtual void setMinThreshold(double minthreshold);

};

/** @brief default predict collector that trace minimal distance with treshhold checking (that is default behavior for most predict logic)
*/
class CV_EXPORTS_W MinDistancePredictCollector : public PredictCollector {
private:
    int _label;
    double _dist;
public:
    /** @brief creates new MinDistancePredictCollector with given threshold */
    CV_WRAP MinDistancePredictCollector(double threshold = DBL_MAX) : PredictCollector(threshold) {
        _label = -1;
        _dist = DBL_MAX;
    };
    CV_WRAP bool emit(const int label, const double dist, const int state = 0);
    CV_WRAP bool filter(int* label, double* dist, const int state);
    /** @brief result label, -1 if not found */
    CV_WRAP int getLabel() const;
    /** @brief result distance (confidence) DBL_MAX if not found */
    CV_WRAP double getDist() const;
    /** @brief factory method to create cv-pointers to MinDistancePredictCollector */
    CV_WRAP static Ptr<MinDistancePredictCollector> create(double threshold = DBL_MAX);
};

/**
@brief Collects top N most close predictions
@note Prevent doubling of same label - if one label is occured twice - most closed distance value will be set
*/
class CV_EXPORTS_W TopNPredictCollector : public PredictCollector {
private:
    size_t _size;
    Ptr<std::list<std::pair<int, double> > > _idx;
public:
    CV_WRAP TopNPredictCollector(size_t size = 5, double threshold = DBL_MAX) : PredictCollector(threshold) {
        _size = size;
        _idx = Ptr<std::list<std::pair<int, double> > >(new std::list<std::pair<int, double> >);
    };
    CV_WRAP bool emit(const int label, const double dist, const int state = 0);
    CV_WRAP bool filter(int* label, double* dist, const int state);
    Ptr<std::list<std::pair<int, double> > > getResult();
    CV_WRAP std::vector<std::pair<int, double> > getResultVector(); // pythonable version
    CV_WRAP static Ptr<TopNPredictCollector> create(size_t size = 5, double threshold = DBL_MAX);
};


/**
@brief Collects all predict results to single vector
@note this collector not analyze double labels in emit, it's raw copy of source prediction result,
remember that filter is still applyed so you can use min/max threshold , distanceKoef and excludeLabel
*/
class CV_EXPORTS_W VectorPredictCollector : public PredictCollector {
private:
    Ptr<std::vector<std::pair<int, double> > > _idx;
public:
    CV_WRAP static const int DEFAULT_SIZE = 5; // top 5 by default
    CV_WRAP VectorPredictCollector(double threshold = DBL_MAX) : PredictCollector(threshold) {
        _idx = Ptr<std::vector<std::pair<int, double> > >(new std::vector<std::pair<int, double> >);
    }
    CV_WRAP bool emit(const int label, const double dist, const int state = 0);
    Ptr<std::vector<std::pair<int, double> > > getResult();
    CV_WRAP std::vector<std::pair<int, double> > getResultVector(); // pythonable version
    CV_WRAP static Ptr<VectorPredictCollector> create(double threshold = DBL_MAX);
};


/**
@brief Collects all predict results to single vector
@note this collector not analyze double labels in emit, it's raw copy of source prediction result,
remember that filter is still applyed so you can use min/max threshold , distanceKoef and excludeLabel
*/
class CV_EXPORTS_W MapPredictCollector : public PredictCollector {
private:
    Ptr<std::map<int, double> > _idx;
public:
    CV_WRAP static const int DEFAULT_SIZE = 5; // top 5 by default
    CV_WRAP MapPredictCollector(double threshold = DBL_MAX) : PredictCollector(threshold) {
        _idx = Ptr<std::map<int, double> >(new std::map<int, double>);
    }
    CV_WRAP bool emit(const int label, const double dist, const int state = 0);
    Ptr<std::map<int, double> > getResult();
    CV_WRAP std::vector<std::pair<int, double> >  getResultVector(); // pythonable version
    CV_WRAP static Ptr<MapPredictCollector> create(double threshold = DBL_MAX);
};

/**
@brief Collects basic statistic information about prediction
@note stat predict collector is usefull for determining valid thresholds
on given trained set, additionally it's required to
evaluate unified koefs between algorithms
*/
class CV_EXPORTS_W StatPredictCollector : public PredictCollector {
private:
    double _min;
    double _max;
    int _count;
    double _sum;
public:
    CV_WRAP StatPredictCollector(double threshold = DBL_MAX) : PredictCollector(threshold) {
        _min = DBL_MAX;
        _max = DBL_MIN;
        _count = 0;
        _sum = 0;
    }
    CV_WRAP bool emit(const int label, const double dist, const int state = 0);
    CV_WRAP double getMin();
    CV_WRAP double getMax();
    CV_WRAP double getSum();
    CV_WRAP int getCount();
    CV_WRAP static Ptr<StatPredictCollector> create(double threshold = DBL_MAX);
};

/**
@brief evaluates standard deviation of given prediction session over trained set
@note in combine with StatPredictCollector can provide statistically based metrices
for thresholds
*/
class CV_EXPORTS_W StdPredictCollector : public PredictCollector {
private:
    double _avg;
    double _n;
    double _s;
public:
    CV_WRAP StdPredictCollector(double threshold = DBL_MAX, double avg = 0) : PredictCollector(threshold) {
        _avg = avg;
        _n = 0;
        _s = 0;
    }
    CV_WRAP bool emit(const int label, const double dist, const int state = 0);
    CV_WRAP double getResult();
    CV_WRAP static Ptr<StdPredictCollector> create(double threshold = DBL_MAX, double avg = 0);
};



//! @}
}
}

#endif