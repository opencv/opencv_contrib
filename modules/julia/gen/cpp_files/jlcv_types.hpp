// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2020 by Archit Rungta

template<typename T>
struct CxxPoint
{
  T x;
  T y;
};
template<typename T>
struct CxxPoint3
{
  T x;
  T y;
  T z;
};

template<typename T>
struct CxxSize
{
  T width;
  T height;
};

template<typename T>
struct CxxRect
{
  T x;
  T y;
  T width;
  T height;
};

struct CxxRotatedRect
{
    Point2f center;
    Size2f size;
    float angle;
};

struct CxxRange
{
    int start;
    int end;
};

struct CxxTermCriteria
{
    int type;
    int maxCount;
    double epsilon;
};


template<typename T>
struct CxxComplex
{
  T re;
  T im;
};


namespace jlcxx
{
  template <> struct IsMirroredType<cv::Range> : std::true_type {};
  template <> struct IsMirroredType<cv::RotatedRect> : std::true_type {};
  template <> struct IsMirroredType<cv::TermCriteria> : std::true_type {};

  template<typename T> struct IsMirroredType<cv::Point_<T>> : std::true_type {};

  template<typename T> struct static_type_mapping<cv::Point_<T>> { using type = CxxPoint<T>; };

  template<typename T>
  struct julia_type_factory<cv::Point_<T>>
  {
    static inline jl_datatype_t* julia_type()
    {
      return (jl_datatype_t*)apply_type((jl_value_t*)jlcxx::julia_type("Point"), jl_svec1(julia_base_type<T>()));
    }
  };

  template<typename T>
  struct ConvertToJulia<cv::Point_<T>, NoMappingTrait>
  {
    CxxPoint<T> operator()(const cv::Point_<T>& cpp_val) const
    {
      return *reinterpret_cast<const CxxPoint<T>*>(&cpp_val);
    }
  };
  template<typename T>
  struct ConvertToCpp<cv::Point_<T>, NoMappingTrait>
  {
    inline cv::Point operator()(const CxxPoint<T>& julia_val) const
    {
      return *reinterpret_cast<const cv::Point_<T>*>(&julia_val);
    }
  };



  template<typename T> struct IsMirroredType<cv::Size_<T>> : std::true_type {};

  template<typename T> struct static_type_mapping<cv::Size_<T>> { using type = CxxSize<T>; };

  template<typename T>
  struct julia_type_factory<cv::Size_<T>>
  {
    static inline jl_datatype_t* julia_type()
    {
      return (jl_datatype_t*)apply_type((jl_value_t*)jlcxx::julia_type("Size"), jl_svec1(julia_base_type<T>()));
    }
  };

  template<typename T>
  struct ConvertToJulia<cv::Size_<T>, NoMappingTrait>
  {
    CxxSize<T> operator()(const cv::Size_<T>& cpp_val) const
    {
      return *reinterpret_cast<const CxxSize<T>*>(&cpp_val);
    }
  };
  template<typename T>
  struct ConvertToCpp<cv::Size_<T>, NoMappingTrait>
  {
    inline cv::Size operator()(const CxxSize<T>& julia_val) const
    {
      return *reinterpret_cast<const cv::Size_<T>*>(&julia_val);
    }
  };




  template<typename T> struct IsMirroredType<cv::Point3_<T>> : std::true_type {};

  template<typename T> struct static_type_mapping<cv::Point3_<T>> { using type = CxxPoint3<T>; };

  template<typename T>
  struct julia_type_factory<cv::Point3_<T>>
  {
    static inline jl_datatype_t* julia_type()
    {
      return (jl_datatype_t*)apply_type((jl_value_t*)jlcxx::julia_type("Point3"), jl_svec1(julia_base_type<T>()));
    }
  };

  template<typename T>
  struct ConvertToJulia<cv::Point3_<T>, NoMappingTrait>
  {
    CxxPoint3<T> operator()(const cv::Point3_<T>& cpp_val) const
    {
      return *reinterpret_cast<const CxxPoint3<T>*>(&cpp_val);
    }
  };
  template<typename T>
  struct ConvertToCpp<cv::Point3_<T>, NoMappingTrait>
  {
    inline cv::Point3_<T> operator()(const CxxPoint3<T>& julia_val) const
    {
      return *reinterpret_cast<const cv::Point3_<T>*>(&julia_val);
    }
  };

  template<typename T> struct IsMirroredType<cv::Rect_<T>> : std::true_type {};

  template<typename T> struct static_type_mapping<cv::Rect_<T>> { using type = CxxRect<T>; };

  template<typename T>
  struct julia_type_factory<cv::Rect_<T>>
  {
    static inline jl_datatype_t* julia_type()
    {
      return (jl_datatype_t*)apply_type((jl_value_t*)jlcxx::julia_type("Rect"), jl_svec1(julia_base_type<T>()));
    }
  };

  template<typename T>
  struct ConvertToJulia<cv::Rect_<T>, NoMappingTrait>
  {
    CxxRect<T> operator()(const cv::Rect_<T>& cpp_val) const
    {
      return *reinterpret_cast<const CxxRect<T>*>(&cpp_val);
    }
  };
  template<typename T>
  struct ConvertToCpp<cv::Rect_<T>, NoMappingTrait>
  {
    inline cv::Rect operator()(const CxxRect<T>& julia_val) const
    {
      return *reinterpret_cast<const cv::Rect_<T>*>(&julia_val);
    }
  };


  template<typename T> struct IsMirroredType<cv::Complex<T>> : std::true_type {};

  template<typename T> struct static_type_mapping<cv::Complex<T>> { using type = CxxComplex<T>; };

  template<typename T>
  struct julia_type_factory<cv::Complex<T>>
  {
    static inline jl_datatype_t* julia_type()
    {
      return (jl_datatype_t*)apply_type((jl_value_t*)jlcxx::julia_type("cvComplex"), jl_svec1(julia_base_type<T>()));
    }
  };

  template<typename T>
  struct ConvertToJulia<cv::Complex<T>, NoMappingTrait>
  {
    CxxComplex<T> operator()(const cv::Complex<T>& cpp_val) const
    {
      return *reinterpret_cast<const CxxComplex<T>*>(&cpp_val);
    }
  };
  template<typename T>
  struct ConvertToCpp<cv::Complex<T>, NoMappingTrait>
  {
    inline cv::Complex<T> operator()(const CxxComplex<T>& julia_val) const
    {
      return *reinterpret_cast<const cv::Complex<T>*>(&julia_val);
    }
  };

template<typename T>
struct BoxValue<Size_<T>,CxxSize<T>>
{
  inline jl_value_t* operator()(Size_<T> cppval)
  {
    return jl_new_bits((jl_value_t*)julia_type<Size_<T>>(), reinterpret_cast<CxxSize<T>*>(&cppval));
  }

  inline jl_value_t* operator()(Size_<T> cppval, jl_value_t* dt)
  {
    return jl_new_bits(dt, reinterpret_cast<CxxSize<T>*>(&cppval));
  }
};

template<typename T>
struct BoxValue<Point_<T>,CxxPoint<T>>
{
  inline jl_value_t* operator()(Point_<T> cppval)
  {
    return jl_new_bits((jl_value_t*)julia_type<Point_<T>>(), reinterpret_cast<CxxPoint<T>*>(&cppval));
  }

  inline jl_value_t* operator()(Point_<T> cppval, jl_value_t* dt)
  {
    return jl_new_bits(dt, reinterpret_cast<CxxPoint<T>*>(&cppval));
  }
};

template<typename T>
struct BoxValue<Point3_<T>,CxxPoint3<T>>
{
  inline jl_value_t* operator()(Point3_<T> cppval)
  {
    return jl_new_bits((jl_value_t*)julia_type<Point3_<T>>(), reinterpret_cast<CxxPoint3<T>*>(&cppval));
  }

  inline jl_value_t* operator()(Point3_<T> cppval, jl_value_t* dt)
  {
    return jl_new_bits(dt, reinterpret_cast<CxxPoint3<T>*>(&cppval));
  }
};

template<typename T>
struct BoxValue<Rect_<T>,CxxRect<T>>
{
  inline jl_value_t* operator()(Rect_<T> cppval)
  {
    return jl_new_bits((jl_value_t*)julia_type<Rect_<T>>(), reinterpret_cast<CxxRect<T>*>(&cppval));
  }

  inline jl_value_t* operator()(Rect_<T> cppval, jl_value_t* dt)
  {
    return jl_new_bits(dt, reinterpret_cast<CxxRect<T>*>(&cppval));
  }
};

};