//| This file is a part of the sferes2 framework.
//| Copyright 2009, ISIR / Universite Pierre et Marie Curie (UPMC)
//| Main contributor(s): Jean-Baptiste Mouret, mouret@isir.fr
//|
//| This software is a computer program whose purpose is to facilitate
//| experiments in evolutionary computation and evolutionary robotics.
//|
//| This software is governed by the CeCILL license under French law
//| and abiding by the rules of distribution of free software.  You
//| can use, modify and/ or redistribute the software under the terms
//| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
//| following URL "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and rights to
//| copy, modify and redistribute granted by the license, users are
//| provided only with a limited warranty and the software's author,
//| the holder of the economic rights, and the successive licensors
//| have only limited liability.
//|
//| In this respect, the user's attention is drawn to the risks
//| associated with loading, using, modifying and/or developing or
//| reproducing the software by the user in light of its specific
//| status of free software, that may mean that it is complicated to
//| manipulate, and that also therefore means that it is reserved for
//| developers and experienced professionals having in-depth computer
//| knowledge. Users are therefore encouraged to load and test the
//| software's suitability as regards their requirements in conditions
//| enabling the security of their systems and/or data to be ensured
//| and, more generally, to use and operate it in the same conditions
//| as regards security.
//|
//| The fact that you are presently reading this means that you have
//| had knowledge of the CeCILL license and that you accept its terms.




#ifndef STC_HPP
#define STC_HPP
namespace stc {

  template <class Exact>
  class Any {
  };

  template <typename Exact>
  Exact& exact(Any<Exact>& ref) {
    return *(Exact*)(void*)(&ref);
  }

  template <typename Exact>
  const Exact& exact(const Any<Exact>& cref) {
    return *(const Exact*)(const void*)(&cref);
  }

  template <typename Exact>
  Exact* exact(Any<Exact>* ptr) {
    return (Exact*)(void*)(ptr);
  }

  template <typename Exact>
  const Exact* exact(const Any<Exact>* cptr) {
    return (const Exact*)(const void*)(cptr);
  }

  struct Itself {};

  // default version
  template <class T, class Exact>
  struct FindExact {
    typedef Exact ret;
  };
  // version specialized for Exact=Itself
  template <class T>
  struct FindExact<T, Itself> {
    typedef T ret;
  };
  struct _Params {};

}

#define STC_FIND_EXACT(Type) typename stc::FindExact<Type<Exact>, Exact>::ret


// eq. class Class
#define STC_CLASS(Class)			\
  template<typename Exact = stc::Itself>	\
  class Class : public stc::Any<Exact>



// eq. class Class1 : public Parent
#define STC_CLASS_D(Class, Parent)			\
  template <typename Exact = stc::Itself>		\
  class Class : public Parent<STC_FIND_EXACT(Class)>



// return the parent class (eq. Class2)
#define STC_PARENT(Class, Parent) Parent<STC_FIND_EXACT(Class)>


// eq. class Class
#define SFERES_CLASS(Class)					\
  template<typename Params = stc::_Params, typename Exact = stc::Itself> \
  class Class : public stc::Any<Exact>

// eq. class Class1 : public Parent
#define SFERES_CLASS_D(Class, Parent)					\
  template <typename Params = stc::_Params, typename Exact = stc::Itself> \
  class Class : public Parent<Params,  typename stc::FindExact<Class<Params, Exact>, Exact>::ret>

// to call the parent constructor
#define SFERES_PARENT(Class, Parent) Parent<Params,  typename stc::FindExact<Class<Params, Exact>, Exact>::ret>



// to simulate a static array (to be used in Param)
// from : SFERES_ARRAY(my_type, my_name, 0.2, 0.4)
// this generates 2 functions :
// - my_type my_name(size_t i)
// - size_t my_name_size()
// and a typedef my_type my_name_t
#define SFERES_ARRAY(T, A, ...)						\
  static const T A(size_t i)						\
  { assert(i < A##_size()); SFERES_CONST T _##A[] = { __VA_ARGS__ }; return _##A[i]; } \
  static const size_t A##_size()					\
  { SFERES_CONST T _##A[] = { __VA_ARGS__ }; return sizeof(_##A) / sizeof(T); } \
  typedef T A##_t;

// to simulate a string (to be used in Param)
#define SFERES_STRING(N, V) static const char* N() { return V; }


#define SFERES_CONST BOOST_STATIC_CONSTEXPR

#endif
