#ifndef __OPENCV_DNN_DNN_DICT_HPP__
#define __OPENCV_DNN_DNN_DICT_HPP__

#include <opencv2/core.hpp>
#include <map>
#include <ostream>

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

    template<typename TypeIter>
    static DictValue arrayInt(TypeIter begin, int size);
    template<typename TypeIter>
    static DictValue arrayReal(TypeIter begin, int size);
    template<typename TypeIter>
    static DictValue arrayString(TypeIter begin, int size);

    template<typename T>
    T get(int idx = -1) const;

    int size() const;

    bool isInt() const;
    bool isString() const;
    bool isReal() const;

    DictValue &operator=(const DictValue &r);

    friend std::ostream &operator<<(std::ostream &stream, const DictValue &dictv);

    ~DictValue();

protected:

    int type;

    union
    {
        AutoBuffer<int64, 1> *pi;
        AutoBuffer<double, 1> *pd;
        AutoBuffer<String, 1> *ps;
        void *p;
    };

    DictValue(int _type, void *_p) : type(_type), p(_p) {}
    void release();
};

template<typename TypeIter>
DictValue DictValue::arrayInt(TypeIter begin, int size)
{
    DictValue res(Param::INT, new AutoBuffer<int64, 1>(size));
    for (int j = 0; j < size; begin++, j++)
        (*res.pi)[j] = *begin;
    return res;
}

template<typename TypeIter>
DictValue DictValue::arrayReal(TypeIter begin, int size)
{
    DictValue res(Param::REAL, new AutoBuffer<double, 1>(size));
    for (int j = 0; j < size; begin++, j++)
        (*res.pd)[j] = *begin;
    return res;
}

template<typename TypeIter>
DictValue DictValue::arrayString(TypeIter begin, int size)
{
    DictValue res(Param::STRING, new AutoBuffer<String, 1>(size));
    for (int j = 0; j < size; begin++, j++)
        (*res.ps)[j] = *begin;
    return res;
}

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

    const DictValue &get(const String &name) const
    {
        _Dict::const_iterator i = dict.find(name);
        if (i == dict.end())
            CV_Error(Error::StsBadArg, "Required argument \"" + name + "\" not found into dictionary");
        return i->second;
    }

    template <typename T>
    T get(const String &name) const
    {
        return this->get(name).get<T>();
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

    friend std::ostream &operator<<(std::ostream &stream, const Dict &dict);
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
    CV_Assert(idx == -1 && size() == 1 || idx >= 0 && idx < size());
    idx = (idx == -1) ? 0 : idx;

    if (type == Param::INT)
    {
        return (*pi)[idx];
    }
    else if (type == Param::REAL)
    {
        double doubleValue = (*pd)[idx];

        double fracpart, intpart;
        fracpart = std::modf(doubleValue, &intpart);
        CV_Assert(fracpart == 0.0);

        return (int64)doubleValue;
    }
    else
    {
        CV_Assert(isInt() || isReal());
        return 0;
    }
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
    CV_Assert(idx == -1 && size() == 1 || idx >= 0 && idx < size());
    idx = (idx == -1) ? 0 : idx;

    if (type == Param::REAL)
    {
        return (*pd)[idx];
    }
    else if (type == Param::INT)
    {
        return (double)(*pi)[idx];
    }
    else
    {
        CV_Assert(isReal() || isInt());
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
    CV_Assert(isString());
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
    type = r.type;

    if (r.type == Param::INT)
        pi = new AutoBuffer<int64, 1>(*r.pi);
    else if (r.type == Param::STRING)
        ps = new AutoBuffer<String, 1>(*r.ps);
    else if (r.type == Param::REAL)
        pd = new AutoBuffer<double, 1>(*r.pd);
}

inline bool DictValue::isString() const
{
    return (type == Param::STRING);
}

inline bool DictValue::isInt() const
{
    return (type == Param::INT);
}

inline bool DictValue::isReal() const
{
    return (type == Param::REAL || type == Param::INT);
}

inline int DictValue::size() const
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
        CV_Error(Error::StsInternal, "");
        return -1;
    }
}

inline std::ostream &operator<<(std::ostream &stream, const DictValue &dictv)
{
    int i;

    if (dictv.isInt())
    {
        for (i = 0; i < dictv.size() - 1; i++)
            stream << dictv.get<int64>(i) << ", ";
        stream << dictv.get<int64>(i);
    }
    else if (dictv.isReal())
    {
        for (i = 0; i < dictv.size() - 1; i++)
            stream << dictv.get<double>(i) << ", ";
        stream << dictv.get<double>(i);
    }
    else if (dictv.isString())
    {
        for (i = 0; i < dictv.size() - 1; i++)
            stream << "\"" << dictv.get<String>(i) << "\", ";
        stream << dictv.get<String>(i);
    }

    return stream;
}

inline std::ostream &operator<<(std::ostream &stream, const Dict &dict)
{
    Dict::_Dict::const_iterator it;
    for (it = dict.dict.begin(); it != dict.dict.end(); it++)
        stream << it->first << " : " << it->second << "\n";

    return stream;
}

}
}

#endif
