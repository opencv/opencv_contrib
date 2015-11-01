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

#ifndef _NN_PARAMS_HPP_
#define _NN_PARAMS_HPP_
namespace nn
{
  namespace params
  {
    struct Dummy
    {
      friend std::ostream& operator<<(std::ostream& output, const Dummy& e);
      Dummy() : _x(42) {}
      void mutate() {}
      void random() {}
      void develop() {}
      size_t size() const { return 0; }
      float data(size_t i) const { return 0.0f;}
      float& operator[](size_t i) { return _x; }
      float operator[](size_t i) const { return _x; }
      template<typename A>
      void serialize(A& ar, unsigned int v) {}
      typedef float type_t;
    protected:
      float _x;
    };

    std::ostream& operator<<(std::ostream& output, const Dummy& e) {
      return output;
    }
    template<int S>
    struct Vectorf
    {
      typedef float type_t;
      BOOST_STATIC_CONSTEXPR int s=S;
      Vectorf() : _data(S) {}
      // magic cast !
      template<typename T>
      Vectorf(const T& v) :
	_data(S)
      {
      	assert(v.size() == S);
      	for (size_t i = 0; i < v.size(); ++i)
      	  _data[i] = v.data(i);
      }
      float data(size_t i) const { assert(i < S) ; return _data[i]; }
      float& operator[](size_t i) { return _data[i]; }
      float operator[](size_t i) const { return _data[i]; }

      size_t size() const { return _data.size(); }
      void mutate() {}
      void random() {}
      void develop() {}
    protected:
      std::vector<float> _data;
    };
  }



}
#endif
