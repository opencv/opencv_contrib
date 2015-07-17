#ifndef __OPENCV_DNN_DNN_DICT_HPP__
#define __OPENCV_DNN_DNN_DICT_HPP__

#include <opencv2/core.hpp>
#include <map>

namespace cv
{
namespace dnn
{

struct DictValue
{
    DictValue(const DictValue &r);
    DictValue(int p = 0)        : type(Param::INT), pi(new AutoBuffer<int64,1>) { (*pi)[0] = p; }
    DictValue(unsigned p)       : type(Param::INT), pi(new AutoBuffer<int64,1>) { (*pi)[0] = p; }
    DictValue(double p)         : type(Param::REAL), pd(new AutoBuffer<double,1>) { (*pd)[0] = p; }
    DictValue(const String &p)  : type(Param::STRING), ps(new AutoBuffer<String,1>) { (*ps)[0] = p; }

    template<typename T>
    T get(int idx = -1) const;

    int size() const;

    bool isInt() const;
    bool isString() const;
    bool isReal() const;    

    DictValue &operator=(const DictValue &r);

    ~DictValue();

protected:

    int type;

    union
    {
        AutoBuffer<int64, 1> *pi;
        AutoBuffer<double, 1> *pd;
        AutoBuffer<String, 1> *ps;
    };

    void release();
};

class CV_EXPORTS Dict
{
    typedef std::map<String, DictValue> _Dict;
    _Dict dict;

public:

    bool has(const String &name)
    {
        return dict.count(name) != 0;
    }

    DictValue *ptr(const String &name)
    {
        _Dict::iterator i = dict.find(name);
        return (i == dict.end()) ? NULL : &i->second;
    }

    template <typename T>
    T get(const String &name) const
    {
        _Dict::const_iterator i = dict.find(name);
        if (i == dict.end())
            CV_Error(cv::Error::StsBadArg, "Required argument \"" + name + "\" not found into dictionary");
        return i->second.get<T>();
    }

    template <typename T>
    T get(const String &name, const T &default_value) const
    {
        _Dict::const_iterator i = dict.find(name);

        if (i != dict.end())
            return i->second.get<T>();
        else
            return default_value;
    }

    template<typename T>
    const T &set(const String &name, const T &value)
    {
        _Dict::iterator i = dict.find(name);

        if (i != dict.end())
            i->second = DictValue(value);
        else
            dict.insert(std::make_pair(name, DictValue(value)));

        return value;
    }
};

template<>
inline DictValue DictValue::get<DictValue>(int idx) const
{
    CV_Assert(idx == -1);
    return *this;
}

template<>
inline int64 DictValue::get<int64>(int idx) const
{
    CV_Assert(type == Param::INT);
    CV_Assert(idx == -1 && pi->size() == 1 || idx >= 0 && idx < (int)pi->size());
    return (*pi)[(idx == -1) ? 0 : idx];
}

template<>
inline int DictValue::get<int>(int idx) const
{
    return (int)get<int64>(idx);
}

template<>
inline unsigned DictValue::get<unsigned>(int idx) const
{
    return (unsigned)get<int64>(idx);
}

template<>
inline bool DictValue::get<bool>(int idx) const
{
    return (get<int64>(idx) != 0);
}

template<>
inline double DictValue::get<double>(int idx) const
{
    if (type == Param::REAL)
    {
        CV_Assert(idx == -1 && pd->size() == 1 || idx >= 0 && idx < (int)pd->size());
        return (*pd)[0];
    }
    else if (type == Param::INT)
    {
        CV_Assert(idx == -1 && pi->size() == 1 || idx >= 0 && idx < (int)pi->size());
        return (double)(*pi)[0];;
    }
    else
    {
        CV_Assert(type == Param::REAL || type == Param::INT);
        return 0;
    }
}

template<>
inline float DictValue::get<float>(int idx) const
{
    return (float)get<double>(idx);
}

template<>
inline String DictValue::get<String>(int idx) const
{
    CV_Assert(type == Param::STRING);
    CV_Assert(idx == -1 && ps->size() == 1 || idx >= 0 && idx < (int)ps->size());
    return (*ps)[(idx == -1) ? 0 : idx];
}

inline void DictValue::release()
{
    switch (type)
    {
    case Param::INT:
        delete pi;
        break;
    case Param::STRING:
        delete ps;
        break;
    case Param::REAL:
        delete pd;
        break;
    }
}

inline DictValue::~DictValue()
{
    release();
}

inline DictValue & DictValue::operator=(const DictValue &r)
{
    if (&r == this)
        return *this;

    if (r.type == Param::INT)
    {
        AutoBuffer<int64, 1> *tmp = new AutoBuffer<int64, 1>(*r.pi);
        release();
        pi = tmp;
    }
    else if (r.type == Param::STRING)
    {
        AutoBuffer<String, 1> *tmp = new AutoBuffer<String, 1>(*r.ps);
        release();
        ps = tmp;
    }
    else if (r.type == Param::REAL)
    {
        AutoBuffer<double, 1> *tmp = new AutoBuffer<double, 1>(*r.pd);
        release();
        pd = tmp;
    }

    type = r.type;

    return *this;
}

inline DictValue::DictValue(const DictValue &r)
{
    *this = r;
}

inline bool DictValue::isString() const
{
    return (type == Param::STRING);
}

inline bool DictValue::isInt() const
{
    return (type == Param::INT);
}

bool DictValue::isReal() const
{
    return (type == Param::REAL);
}

int DictValue::size() const
{
    switch (type)
    {
    case Param::INT:
        return (int)pi->size();
        break;
    case Param::STRING:
        return (int)ps->size();
        break;
    case Param::REAL:
        return (int)pd->size();
        break;
    default:
        return -1;
    }
}

}
}

#endif
