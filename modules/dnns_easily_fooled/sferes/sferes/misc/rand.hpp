

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




#ifndef RAND_HPP_
#define RAND_HPP_

#include <cstdlib>
#include <cmath>
#include <list>
#include <stdlib.h>
#include <boost/swap.hpp>
#include <boost/random.hpp>

// someday we will have a real thread-safe random number generator...
namespace sferes {
  namespace misc {
    // NOT Thread-safe !
    template<typename T>
    inline T rand(T max = 1.0) {
      assert(max > 0);
      T v;
      do
        v = T(((double)max * ::rand())/(RAND_MAX + 1.0));
      while(v >= max); // this strange case happened... precision problem?
      assert(v < max);
      return v;
    }


    template<typename T>
    inline T rand(T min, T max) {
      assert(max != min);
      assert(max > min);
      T res = T(rand<double>() * ((long int) max - (long int) min) + min);
      assert(res >= min);
      assert(res < max);
      return res;
    }

    template<typename T>
    inline T gaussian_rand(T m=0.0,T  v=1.0) {
      float facteur = sqrt(-2.0f * log(rand<float>()));
      float trigo  = 2.0f * M_PI * rand<float>();

      return T(m + v * facteur * cos(trigo));

    }

    inline void rand_ind(std::vector<size_t>& a1, size_t size) {
      a1.resize(size);
      for (size_t i = 0; i < a1.size(); ++i)
        a1[i] = i;
      for (size_t i = 0; i < a1.size(); ++i) {
        size_t k = rand(i, a1.size());
        assert(k < a1.size());
        boost::swap(a1[i], a1[k]);
      }
    }


    /// return a random it in the list
    template<typename T>
    inline typename std::list<T>::iterator rand_in_list(std::list<T>& l) {
      int n = rand(l.size());
      typename std::list<T>::iterator it = l.begin();
      for (int i = 0; i < n; ++i)
        ++it;
      return it;
    }


    inline bool flip_coin() {
      return rand<float>() < 0.5f;
    }

    template<typename L>
    inline typename L::iterator rand_l(L& l) {
      size_t k = rand(l.size());
      typename L::iterator it = l.begin();
      for (size_t i = 0; i < k; ++i)
        ++it;
      return it;
    }
  }
}
#endif
