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




#ifndef GEN_CMAES_HPP_
#define GEN_CMAES_HPP_

#ifdef EIGEN3_ENABLED

#include <iostream>
#include <cmath>
#include <limits>

#include <boost/foreach.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/static_assert.hpp>

#include <Eigen/Core>
#include <Eigen/QR>

#include <sferes/dbg/dbg.hpp>
#include <sferes/stc.hpp>
#include <sferes/misc.hpp>


namespace sferes {
  namespace gen {
    // this class requires EIGEN3 (libEIGEN3-dev)
    // REFERENCE:
    // Hansen, N. and S. Kern (2004). Evaluating the CMA Evolution
    // Strategy on Multimodal Test Functions.  Eighth International
    // Conference on Parallel Problem Solving from Nature PPSN VIII,
    // Proceedings, pp. 282-291, Berlin: Springer.
    // (http://www.bionik.tu-berlin.de/user/niko/ppsn2004hansenkern.pdf)
    template<int Size, typename Params, typename Exact = stc::Itself>
    class Cmaes : public stc::Any<Exact> {
     public:
      typedef Params params_t;
      typedef Cmaes<Size, Params, Exact> this_t;
      typedef Eigen::Matrix<float, Size, 1> vector_t;
      typedef Eigen::Matrix<float, Size, Size> matrix_t;
      SFERES_CONST size_t es_size = Size;
      Cmaes() : _arx(vector_t::Zero()) { }

      void random() {
      }
      void mutate(const vector_t& xmean,
                  float sigma,
                  const matrix_t& B,
                  const matrix_t& D) {
        for (size_t i = 0; i < Size; ++i)
          _arz[i] = misc::gaussian_rand<float>();
        _arx = xmean + sigma * (B * D * _arz);
      }

      float data(size_t i) const {
        assert(i < _arx.size());
        return _arx[i];
      }
      const vector_t& data() const {
        return _arx;
      }
      const vector_t& arx() const {
        return _arx;
      }
      const vector_t& arz() const {
        return _arz;
      }

      size_t size() const {
        return Size;
      }

      template<typename Archive>
      void save(Archive& a, const unsigned version) const {
        std::vector<float> v(Size);
        for (size_t i = 0; i < Size; ++i)
          v[i] = _arx[i];
        a & BOOST_SERIALIZATION_NVP(v);
      }
      template<typename Archive>
      void load(Archive& a, const unsigned version) {
        std::vector<float> v;
        a & BOOST_SERIALIZATION_NVP(v);
        assert(v.size() == Size);
        for (size_t i = 0; i < Size; ++i)
          _arx[i] = v[i];
      }
      BOOST_SERIALIZATION_SPLIT_MEMBER();
     protected:
      vector_t _arx, _arz;
    };
  } // gen
} // sferes

#else
#warning Eigen3 is disabled -> no CMAES
#endif

#endif
