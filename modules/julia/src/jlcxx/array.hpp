// This file is a modified array.hpp from https://github.com/JuliaInterop/libcxxwrap-julia
// required for the hack that allows automated conversion of OpenCV types.
// Shouldn't be needed once CxxWrap gets inbuilt support
// Here is the original copyright and the license:
/*
==

Copyright (c) 2015: Bart Janssens.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

==
*/


#ifndef JLCXX_ARRAY_HPP
#define JLCXX_ARRAY_HPP

#include "jlcxx/type_conversion.hpp"
#include "jlcxx/tuple.hpp"

namespace jlcxx
{

template<typename PointedT, typename CppT>
struct ValueExtractor
{
  inline CppT operator()(PointedT* p)
  {
    return convert_to_cpp<CppT>(*p);
  }
};


template<typename PointedT>
struct ValueExtractor<PointedT, PointedT>
{
  inline PointedT& operator()(PointedT* p)
  {
    return *p;
  }
};

template<typename PointedT, typename CppT>
class array_iterator_base : public std::iterator<std::random_access_iterator_tag, CppT>
{
private:
  PointedT* m_ptr;
public:
  array_iterator_base() : m_ptr(nullptr)
  {
  }

  explicit array_iterator_base(PointedT* p) : m_ptr(p)
  {
  }

  template <class OtherPointedT, class OtherCppT>
  array_iterator_base(array_iterator_base<OtherPointedT, OtherCppT> const& other) : m_ptr(other.m_ptr) {}

  auto operator*() -> decltype(ValueExtractor<PointedT,CppT>()(m_ptr))
  {
    return ValueExtractor<PointedT,CppT>()(m_ptr);
  }

  array_iterator_base<PointedT, CppT>& operator++()
  {
    ++m_ptr;
    return *this;
  }

  array_iterator_base<PointedT, CppT>& operator--()
  {
    --m_ptr;
    return *this;
  }

  array_iterator_base<PointedT, CppT>& operator+=(std::ptrdiff_t n)
  {
    m_ptr += n;
    return *this;
  }

  array_iterator_base<PointedT, CppT>& operator-=(std::ptrdiff_t n)
  {
    m_ptr -= n;
    return *this;
  }

  PointedT* ptr() const
  {
    return m_ptr;
  }
};

/// Wrap a Julia 1D array in a C++ class. Array is allocated on the C++ side
template<typename ValueT>
class Array
{
public:
  Array(const size_t n = 0)
  {
    jl_value_t* array_type = apply_array_type(julia_type<ValueT>(), 1);
    m_array = jl_alloc_array_1d(array_type, n);
  }

  Array(jl_datatype_t* applied_type, const size_t n = 0)
  {
    jl_value_t* array_type = apply_array_type(applied_type, 1);
    m_array = jl_alloc_array_1d(array_type, n);
  }

  /// Append an element to the end of the list
  template<typename VT>
  void push_back(VT&& val)
  {
    JL_GC_PUSH1(&m_array);
    const size_t pos = jl_array_len(m_array);
    jl_array_grow_end(m_array, 1);
    jl_arrayset(m_array, box<ValueT>(val), pos);
    JL_GC_POP();
  }

  /// Access to the wrapped array
  jl_array_t* wrapped()
  {
    return m_array;
  }

  // access to the pointer for GC macros
  jl_array_t** gc_pointer()
  {
    return &m_array;
  }

private:
  jl_array_t* m_array;
};

namespace detail
{

template<typename T, typename TraitT=mapping_trait<T>>
struct ArrayElementType
{
  using type = static_julia_type<T>;
};

template<typename T>
struct ArrayElementType<T,WrappedPtrTrait>
{
  using type = T;
};

}

/// Reference a Julia array in an STL-compatible wrapper
template<typename ValueT, int Dim = 1>
class ArrayRef
{
public:

  using julia_t = typename detail::ArrayElementType<ValueT>::type;

  ArrayRef(jl_array_t* arr) : m_array(arr)
  {
    assert(wrapped() != nullptr);
  }

  /// Convert from existing C-array (memory owned by C++)
  template<typename... SizesT>
  ArrayRef(julia_t* ptr, const SizesT... sizes);

  /// Convert from existing C-array, explicitly setting Julia ownership
  template<typename... SizesT>
  ArrayRef(const bool julia_owned, julia_t* ptr, const SizesT... sizes);

  typedef array_iterator_base<julia_t, ValueT> iterator;
  typedef array_iterator_base<julia_t const, ValueT const> const_iterator;

  inline jl_array_t* wrapped() const
  {
    return m_array;
  }

  iterator begin()
  {
    return iterator(static_cast<julia_t*>(jl_array_data(wrapped())));
  }

  const_iterator begin() const
  {
    return const_iterator(static_cast<julia_t*>(jl_array_data(wrapped())));
  }

  iterator end()
  {
    return iterator(static_cast<julia_t*>(jl_array_data(wrapped())) + jl_array_len(wrapped()));
  }

  const_iterator end() const
  {
    return const_iterator(static_cast<julia_t*>(jl_array_data(wrapped())) + jl_array_len(wrapped()));
  }

  void push_back(const ValueT& val)
  {
    static_assert(Dim == 1, "ArrayRef::push_back is only for 1D ArrayRef");
    static_assert(std::is_same<julia_t,ValueT>::value, "ArrayRef::push_back is only for arrays of fundamental types");
    jl_array_t* arr_ptr = wrapped();
    JL_GC_PUSH1(&arr_ptr);
    const size_t pos = size();
    jl_array_grow_end(arr_ptr, 1);
    jl_arrayset(arr_ptr, box<ValueT>(val), pos);
    JL_GC_POP();
  }

  const julia_t* data() const
  {
    return (julia_t*)jl_array_data(wrapped());
  }

  julia_t* data()
  {
    return (julia_t*)jl_array_data(wrapped());
  }

  std::size_t size() const
  {
    return jl_array_len(wrapped());
  }

  ValueT& operator[](const std::size_t i)
  {
    if constexpr(std::is_same<julia_t, ValueT>::value)
    {
      return data()[i];
    }
    else if constexpr(std::is_same<julia_t, static_julia_type<ValueT>>::value && !std::is_same<julia_t, WrappedCppPtr>::value)
    {
      return *reinterpret_cast<ValueT*>(&data()[i]);
    }
    else
    {
      return *extract_pointer_nonull<ValueT>(data()[i]);
    }
  }

  const ValueT& operator[](const std::size_t i) const
  {
    if constexpr(std::is_same<julia_t, ValueT>::value)
    {
      return data()[i];
    }
     else if constexpr(std::is_same<julia_t, static_julia_type<ValueT>>::value && !std::is_same<julia_t, WrappedCppPtr>::value)
    {
      return *reinterpret_cast<ValueT*>(&data()[i]);
    }
    else
    {
      return *extract_pointer_nonull<ValueT>(data()[i]);
    }
  }

  jl_array_t* m_array;
};

// Conversions
template<typename T, int Dim, typename SubTraitT>
struct static_type_mapping<ArrayRef<T, Dim>, CxxWrappedTrait<SubTraitT>>
{
  typedef jl_array_t* type;
};

namespace detail
{

template<typename T, typename TraitT=mapping_trait<T>>
struct PackedArrayType
{
  static jl_datatype_t* type()
  {
    return julia_type<T>();
  }
};

template<typename T>
struct PackedArrayType<T*, WrappedPtrTrait>
{
  static jl_datatype_t* type()
  {
    return (jl_datatype_t*)apply_type((jl_value_t*)jlcxx::julia_type("Ptr"), jl_svec1(julia_base_type<T>()));
  }
};

template<typename T, typename SubTraitT>
struct PackedArrayType<T,CxxWrappedTrait<SubTraitT>>
{
  static jl_datatype_t* type()
  {
    create_if_not_exists<T&>();
    return julia_type<T&>();
  }
};

}

template<typename T, int Dim>
struct julia_type_factory<ArrayRef<T, Dim>>
{
  static inline jl_datatype_t* julia_type()
  {
    create_if_not_exists<T>();
    return (jl_datatype_t*)apply_array_type(detail::PackedArrayType<T>::type(), Dim);
  }
};

template<typename ValueT, typename... SizesT>
jl_array_t* wrap_array(const bool julia_owned, ValueT* c_ptr, const SizesT... sizes)
{
  jl_datatype_t* dt = julia_type<ArrayRef<ValueT, sizeof...(SizesT)>>();
  jl_value_t *dims = nullptr;
  JL_GC_PUSH1(&dims);
  dims = convert_to_julia(std::make_tuple(static_cast<cxxint_t>(sizes)...));
  jl_array_t* result = jl_ptr_to_array((jl_value_t*)dt, c_ptr, dims, julia_owned);
  JL_GC_POP();
  return result;
}

template<typename ValueT, int Dim>
template<typename... SizesT>
ArrayRef<ValueT, Dim>::ArrayRef(julia_t* c_ptr, const SizesT... sizes) : m_array(wrap_array(false, c_ptr, sizes...))
{
}

template<typename ValueT, int Dim>
template<typename... SizesT>
ArrayRef<ValueT, Dim>::ArrayRef(const bool julia_owned, julia_t* c_ptr, const SizesT... sizes) : m_array(wrap_array(julia_owned, c_ptr, sizes...))
{
}

template<typename ValueT, typename... SizesT>
auto make_julia_array(ValueT* c_ptr, const SizesT... sizes) -> ArrayRef<ValueT, sizeof...(SizesT)>
{
  return ArrayRef<ValueT, sizeof...(SizesT)>(false, c_ptr, sizes...);
}

template<typename T, typename SubTraitT>
struct static_type_mapping<Array<T>, CxxWrappedTrait<SubTraitT>>
{
  typedef jl_array_t* type;
};

template<typename T>
struct julia_type_factory<Array<T>>
{
  static inline jl_datatype_t* julia_type()
  {
    create_if_not_exists<T>();
    return (jl_datatype_t*)apply_array_type(jlcxx::julia_type<T>(), 1);
  }
};

template<typename T, int Dim>
struct ConvertToJulia<ArrayRef<T,Dim>>
{
  template<typename ArrayRefT>
  jl_array_t* operator()(ArrayRefT&& arr) const
  {
    return arr.wrapped();
  }
};

template<typename T>
struct ConvertToJulia<Array<T>>
{
  jl_value_t* operator()(Array<T>&& arr) const
  {
    return (jl_value_t*)arr.wrapped();
  }
};

template<typename T, int Dim, typename SubTraitT>
struct ConvertToCpp<ArrayRef<T,Dim>, CxxWrappedTrait<SubTraitT>>
{
  ArrayRef<T,Dim> operator()(jl_array_t* arr) const
  {
    return ArrayRef<T,Dim>(arr);
  }
};

// Iterator operator implementation
template<typename PointedT, typename CppT>
bool operator!=(const array_iterator_base<PointedT, CppT>& l, const array_iterator_base<PointedT, CppT>& r)
{
  return r.ptr() != l.ptr();
}

template<typename PointedT, typename CppT>
bool operator==(const array_iterator_base<PointedT, CppT>& l, const array_iterator_base<PointedT, CppT>& r)
{
  return r.ptr() == l.ptr();
}

template<typename PointedT, typename CppT>
bool operator<=(const array_iterator_base<PointedT, CppT>& l, const array_iterator_base<PointedT, CppT>& r)
{
  return l.ptr() <= r.ptr();
}

template<typename PointedT, typename CppT>
bool operator>=(const array_iterator_base<PointedT, CppT>& l, const array_iterator_base<PointedT, CppT>& r)
{
  return l.ptr() >= r.ptr();
}

template<typename PointedT, typename CppT>
bool operator>(const array_iterator_base<PointedT, CppT>& l, const array_iterator_base<PointedT, CppT>& r)
{
  return l.ptr() > r.ptr();
}

template<typename PointedT, typename CppT>
bool operator<(const array_iterator_base<PointedT, CppT>& l, const array_iterator_base<PointedT, CppT>& r)
{
  return l.ptr() < r.ptr();
}

template<typename PointedT, typename CppT>
array_iterator_base<PointedT, CppT> operator+(const array_iterator_base<PointedT, CppT>& l, const std::ptrdiff_t n)
{
  return array_iterator_base<PointedT, CppT>(l.ptr() + n);
}

template<typename PointedT, typename CppT>
array_iterator_base<PointedT, CppT> operator+(const std::ptrdiff_t n, const array_iterator_base<PointedT, CppT>& r)
{
  return array_iterator_base<PointedT, CppT>(r.ptr() + n);
}

template<typename PointedT, typename CppT>
array_iterator_base<PointedT, CppT> operator-(const array_iterator_base<PointedT, CppT>& l, const std::ptrdiff_t n)
{
  return array_iterator_base<PointedT, CppT>(l.ptr() - n);
}

template<typename PointedT, typename CppT>
std::ptrdiff_t operator-(const array_iterator_base<PointedT, CppT>& l, const array_iterator_base<PointedT, CppT>& r)
{
  return l.ptr() - r.ptr();
}

}

#endif
