#ifndef __OPENCV_DNN_DICT_HPP__
#define __OPENCV_DNN_DICT_HPP__

#include <opencv2/core.hpp>
#include <map>

namespace cv
{
namespace dnn
{

struct DictValue
{
    int type;

    union
    {
        int64 i;
        double d;
        bool b;
        String *s;
    };

    DictValue(const DictValue &r);
    DictValue(int p = 0)        : type(cv::Param::INT), i(p) {}
    DictValue(unsigned p)       : type(cv::Param::INT), i(p) {}
    DictValue(double p)         : type(cv::Param::REAL), d(p) {}
    DictValue(bool p)           : type(cv::Param::BOOLEAN), b(p) {}
    DictValue(const String &p)  : type(cv::Param::STRING), s(new String(p)) {}
    DictValue(const char *str)  : type(cv::Param::STRING), s(new String(str)) {}

    template<typename T>
    T get() const;

    bool isString() const;
    bool isInt() const;

    DictValue &operator=(const DictValue &r);

    ~DictValue();

private:
    void release();
};

class CV_EXPORTS Dict
{
    //TODO: maybe this mechanism was realized somewhere in OpenCV?
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

    inline void print()
    {
        for (_Dict::const_iterator i = dict.begin(); i != dict.end(); i++)
        {
            std::cout << i->first << std::endl;
        }
    }
};


template<>
inline int DictValue::get<int>() const
{
    CV_Assert(type == cv::Param::INT);
    return (int)i;
}

template<>
inline unsigned DictValue::get<unsigned>() const
{
    CV_Assert(type == cv::Param::INT);
    return (unsigned)i;
}

template<>
inline double DictValue::get<double>() const
{
    if (type == cv::Param::REAL)
        return d;
    else if (type == cv::Param::INT)
        return (double)i;
    else
    {
        CV_Assert(type == cv::Param::REAL || type == cv::Param::INT);
        return 0;
    }
}

template<>
inline float DictValue::get<float>() const
{
    if (type == cv::Param::FLOAT)
        return (float)d;
    else if (type == cv::Param::INT)
        return (float)i;
    else
    {
        CV_Assert(type == cv::Param::FLOAT || type == cv::Param::INT);
        return (float)0;
    }
}

template<>
inline bool DictValue::get<bool>() const
{
    if (type == cv::Param::BOOLEAN)
    {
        return b;
    }
    else if (type == cv::Param::INT)
    {
        return i != 0;
    }
    else
    {
        CV_Assert(type == cv::Param::BOOLEAN || type == cv::Param::INT);
        return 0;
    }
}

template<>
inline String DictValue::get<String>() const
{
    CV_Assert(type == cv::Param::STRING);
    return *s;
}

inline void DictValue::release()
{
    if (type == cv::Param::STRING && s != NULL)
    {
        delete s;
        s = NULL;
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

    if (r.type == cv::Param::STRING)
    {
        String *_s = new String(*r.s);
        release();
        s = _s;
        type = r.type;
    }
    else //flat structure
    {
        //how to copy anonymous union without memcpy?
        for (size_t i = 0; i < sizeof(*this); i++)
            ((uchar*)this)[i] = ((uchar*)&r)[i];
    }

    return *this;
}

inline DictValue::DictValue(const DictValue &r)
{
    *this = r;
}

inline bool DictValue::isString() const
{
    return (type == cv::Param::STRING);
}

inline bool DictValue::isInt() const
{
    return (type == cv::Param::INT);
}

}
}

#endif
